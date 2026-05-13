import os
import cv2
import random
import numpy as np
import argparse
import json
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple


# =========================
# 1. 配置区：别人主要改这里
# =========================
@dataclass
class DatasetConfig:
    # 输入数据根目录
    data_root: str = r'D:/DataSet/multi_data_down'

    # 输入子目录名
    basemap_dir: str = 'basemap'
    src_dir: str = 'src'
    traj_dir: str = 'traj'
    trajpoint_dir: str = 'trajpoint'
    building_label_dir: str = 'building_label'
    map_label_dir: str = os.path.join('map_label', 'label_width2')

    # 输出根目录
    output_root: str = r'D:/DataSet/multi_data_down/train_log_GKS'

    # patch大小
    img_h: int = 256
    img_w: int = 256

    # 滑动窗口参数 (当 stride < img_h/w 时有重叠)
    stride_h: int = 256  # 垂直步长
    stride_w: int = 256  # 水平步长

    # 每张大图采样多少patch（总数是否均分给所有图）- 仅 random 模式有效
    image_num: int = 75

    # 裁剪模式: 'random' 或 'sliding'
    crop_mode: str = 'random'

    # 数据集划分方式
    # train: 在前 split_ratio 区域采样
    # val:   在后 1-split_ratio 区域采样
    split_ratio: float = 7 / 8

    # 随机种子
    random_seed: int = 42

    # 划分比例 (sliding 模式有效)
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # 是否打印详细日志
    verbose: bool = True


# =========================
# 2. 工具函数
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def make_output_dirs(output_root: str):
    sub_dirs = [
        'visualize',
        'src',
        'label',
        'basemap_split',
        'traj_split',
        'trajpoint_split',
        'traj_and_point_split',
        'traj_and_point_and_img_split',
        'src_split',
        'building_label',
        'building_mask_traj_and_point_split',
        'building_mask_PLI_split',
    ]
    for sub in sub_dirs:
        ensure_dir(os.path.join(output_root, sub))


def read_image_checked(path: str, flag):
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(f'无法读取图像，请检查路径是否正确: {path}')
    return img


def save_png(path: str, img: np.ndarray):
    ok = cv2.imwrite(path, img)
    if not ok:
        raise IOError(f'保存失败: {path}')


def save_npy(path: str, arr: np.ndarray):
    np.save(path, arr)


def get_sliding_window_coords(
    x_width: int,
    x_height: int,
    crop_w: int,
    crop_h: int,
    stride_w: int = None,
    stride_h: int = None,
    padding: bool = True
) -> list:
    """
    滑动窗口裁剪坐标生成（带边界填充）

    参数:
        x_width: 原始图像宽度
        x_height: 原始图像高度
        crop_w: 裁剪窗口宽度
        crop_h: 裁剪窗口高度
        stride_w: 水平步长 (默认等于crop_w，即无重叠)
        stride_h: 垂直步长 (默认等于crop_h，即无重叠)
        padding: 边界是否补零填充

    返回:
        [(x, y, need_padding), ...] 列表
        need_padding: True 表示需要padding（边缘情况）
    """
    if stride_w is None:
        stride_w = crop_w
    if stride_h is None:
        stride_h = crop_h

    coords = []

    y = 0
    while y < x_height:
        x = 0
        while x < x_width:
            need_pad_w = (x + crop_w > x_width)
            need_pad_h = (y + crop_h > x_height)

            if padding and (need_pad_w or need_pad_h):
                coords.append((x, y, True))
            elif not padding and (need_pad_w or need_pad_h):
                pass
            else:
                coords.append((x, y, False))

            x += stride_w

        y += stride_h

    return coords


def crop_with_padding(img, x, y, crop_w, crop_h):
    """裁剪图像，边缘用零填充"""
    h, w = img.shape[:2]

    # 计算实际可用的区域
    x1, x2 = x, min(x + crop_w, w)
    y1, y2 = y, min(y + crop_h, h)

    # 提取有效区域
    if len(img.shape) == 3:
        valid_region = img[y1:y2, x1:x2, :]
        # 创建全零的裁剪结果
        result = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
    else:
        valid_region = img[y1:y2, x1:x2]
        result = np.zeros((crop_h, crop_w), dtype=img.dtype)

    # 计算填充位置
    dy = max(0, -y) if y < 0 else 0
    dx = max(0, -x) if x < 0 else 0

    # 填充有效区域
    result[dy:dy+(y2-y1), dx:dx+(x2-x1)] = valid_region

    return result


def get_random_crop_coords(
    x_width: int,
    x_height: int,
    crop_w: int,
    crop_h: int,
    split_ratio: float,
    subset: str = 'train'
) -> Tuple[int, int]:
    """
    根据 subset 决定裁剪区域：
    - train: 从 [0, split_ratio * H) 区域采样
    - val:   从 [split_ratio * H, H) 区域采样
    - all:   全图随机采样
    """
    if x_width <= crop_w or x_height <= crop_h:
        raise ValueError(
            f'原图尺寸过小，无法裁剪。原图尺寸=({x_height}, {x_width}), 裁剪尺寸=({crop_h}, {crop_w})'
        )

    random_width = random.randint(0, x_width - crop_w - 1)

    if subset == 'train':
        max_h = max(0, int(split_ratio * x_height) - crop_h - 1)
        random_height = random.randint(0, max_h)
    elif subset == 'val':
        min_h = int(split_ratio * x_height)
        max_h = x_height - crop_h - 1
        if min_h > max_h:
            raise ValueError('验证集裁剪范围无效，请检查 split_ratio 或图像尺寸。')
        random_height = random.randint(min_h, max_h)
    elif subset == 'all':
        random_height = random.randint(0, x_height - crop_h - 1)
    else:
        raise ValueError(f'未知 subset: {subset}，可选 train / val / all')

    return random_width, random_height


# =========================
# 3. 数据读取与拼接
# =========================
def multi_data_concat(i: int, image_sets: List[str], cfg: DatasetConfig):
    """
    三种模态：
    - basemap: 原始路网数据，单通道
    - src: 原始遥感RGB影像，3通道
    - traj: 轨迹空间特征，单通道
    - traj_point: 轨迹点特征，单通道

    最终 multi_feature 通道顺序：
    [basemap, traj, traj_point, src(BGR三通道)] -> 共6通道
    """

    image_name = image_sets[i]

    basemap_path = os.path.join(cfg.data_root, cfg.basemap_dir, image_name)
    src_path = os.path.join(cfg.data_root, cfg.src_dir, image_name)
    traj_path = os.path.join(cfg.data_root, cfg.traj_dir, image_name)
    trajpoint_path = os.path.join(cfg.data_root, cfg.trajpoint_dir, image_name)
    building_label_path = os.path.join(cfg.data_root, cfg.building_label_dir, image_name)
    map_label_path = os.path.join(cfg.data_root, cfg.map_label_dir, image_name)

    basemap = read_image_checked(basemap_path, cv2.IMREAD_GRAYSCALE)
    src = read_image_checked(src_path, cv2.IMREAD_COLOR)
    traj = read_image_checked(traj_path, cv2.IMREAD_GRAYSCALE)
    traj_point = read_image_checked(trajpoint_path, cv2.IMREAD_GRAYSCALE)
    building_label = read_image_checked(building_label_path, cv2.IMREAD_GRAYSCALE)
    map_label = read_image_checked(map_label_path, cv2.IMREAD_GRAYSCALE)

    # 保持和原始代码一致
    basemap = np.array(basemap, dtype=np.uint8)
    traj = np.array(traj, dtype=np.uint8)
    traj_point = np.array(traj_point, dtype=np.uint8)
    building_label = np.array(building_label, dtype=np.uint8)
    map_label = np.array(map_label, dtype=np.uint8)

    # 增加单通道维度
    basemap = np.expand_dims(basemap, axis=-1)
    traj = np.expand_dims(traj, axis=-1)
    traj_point = np.expand_dims(traj_point, axis=-1)
    map_label = np.expand_dims(map_label, axis=-1)

    # 拼接成6通道特征
    multi_feature = np.dstack((basemap, traj, traj_point, src))

    return multi_feature, map_label, building_label


# =========================
# 4. patch保存逻辑
# =========================
def save_patch_results(
    g_count: int,
    src_roi: np.ndarray,
    label_roi: np.ndarray,
    building_roi: np.ndarray,
    output_root: str
):
    """
    按原始逻辑保存各种中间结果
    src_roi 通道定义：
    0: basemap
    1: traj
    2: traj_point
    3:6: RGB/BGR影像
    """

    visualize = (label_roi.squeeze() * 50).astype(np.uint8)

    # 主数据
    save_png(os.path.join(output_root, 'visualize', f'{g_count}.png'), visualize)
    save_npy(os.path.join(output_root, 'src', f'{g_count}.npy'), src_roi)
    save_png(os.path.join(output_root, 'label', f'{g_count}.png'), label_roi.squeeze().astype(np.uint8))

    # mask相关
    traj_for_mask = src_roi[:, :, 1]
    trajpoint_for_mask = src_roi[:, :, 2]

    building_mask_traj = np.where(building_roi == 2, 0, traj_for_mask)
    building_mask_trajpoint = np.where(building_roi == 2, 0, trajpoint_for_mask)
    building_mask_traj_and_point = np.dstack((building_mask_traj, building_mask_trajpoint))
    building_mask_PLI = np.dstack((building_mask_traj, building_mask_trajpoint, src_roi[:, :, 3:6]))

    # 单独拆分保存
    save_png(os.path.join(output_root, 'basemap_split', f'{g_count}.png'), src_roi[:, :, 0].astype(np.uint8))
    save_png(os.path.join(output_root, 'traj_split', f'{g_count}.png'), src_roi[:, :, 1].astype(np.uint8))
    save_png(os.path.join(output_root, 'trajpoint_split', f'{g_count}.png'), src_roi[:, :, 2].astype(np.uint8))
    save_npy(os.path.join(output_root, 'traj_and_point_split', f'{g_count}.npy'), src_roi[:, :, 1:3])
    save_npy(os.path.join(output_root, 'traj_and_point_and_img_split', f'{g_count}.npy'), src_roi[:, :, 1:6])

    # 这里保持和原始逻辑一致，保存3通道彩色图
    save_png(os.path.join(output_root, 'src_split', f'{g_count}.png'), src_roi[:, :, 3:6].astype(np.uint8))
    save_png(os.path.join(output_root, 'building_label', f'{g_count}.png'), building_roi.astype(np.uint8))
    save_npy(
        os.path.join(output_root, 'building_mask_traj_and_point_split', f'{g_count}.npy'),
        building_mask_traj_and_point
    )
    save_npy(
        os.path.join(output_root, 'building_mask_PLI_split', f'{g_count}.npy'),
        building_mask_PLI
    )


# =========================
# 5. 数据增强接口（占位）
# =========================
def data_augment(src_roi: np.ndarray, label_roi: np.ndarray):
    """
    这里保留接口，按你的原增强函数替换即可
    """
    return src_roi, label_roi


# =========================
# 5.1 保存划分索引到 JSON
# =========================
def save_split_indices(output_root: str, train_indices: List[int], val_indices: List[int], test_indices: List[int]):
    """保存 train/val/test 划分索引到 JSON 文件"""
    split_data = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }
    split_path = os.path.join(output_root, 'split_indices.json')
    with open(split_path, 'w') as f:
        json.dump(split_data, f, indent=2)
    print(f"划分索引已保存到: {split_path}")
    print(f"  train: {len(train_indices)} samples")
    print(f"  val:   {len(val_indices)} samples")
    print(f"  test:  {len(test_indices)} samples")


def random_split_indices(total_count: int, output_root: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    根据 total_count 生成可复现的随机划分索引
    使用固定 seed=42 保证可复现性
    """
    random.seed(42)
    np.random.seed(42)

    indices = list(range(total_count))
    random.shuffle(indices)

    train_end = int(total_count * train_ratio)
    val_end = int(total_count * (train_ratio + val_ratio))

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    save_split_indices(output_root, train_indices, val_indices, test_indices)

    return train_indices, val_indices, test_indices


# =========================
# 6. 主函数：构建数据集
# =========================
def create_dataset(
    image_sets: List[str],
    cfg: DatasetConfig,
    mode: str = 'normal',
    subset: str = 'train'
):
    """
    mode:
        - normal
        - augment
    subset:
        - train
        - val
        - all
    crop_mode:
        - random: 随机裁剪
        - sliding: 滑动窗口裁剪，边缘补零
    """
    print(f'creating dataset... (mode: {cfg.crop_mode})')

    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    make_output_dirs(cfg.output_root)

    num_images = len(image_sets)
    if num_images == 0:
        raise ValueError('image_sets 为空，无法创建数据集。')

    g_count = 0

    for i in tqdm(range(num_images)):
        src_img, label_img, label_building = multi_data_concat(i, image_sets, cfg)
        x_height, x_width, _ = src_img.shape

        if cfg.verbose:
            print(f'[{i+1}/{num_images}] image={image_sets[i]}, shape={src_img.shape}')

        if cfg.crop_mode == 'sliding':
            # 滑动窗口模式
            crop_coords = get_sliding_window_coords(
                x_width=x_width,
                x_height=x_height,
                crop_w=cfg.img_w,
                crop_h=cfg.img_h,
                stride_w=cfg.stride_w,
                stride_h=cfg.stride_h,
                padding=True
            )
        else:
            # 随机裁剪模式
            image_each = max(1, cfg.image_num // num_images)
            crop_coords = []
            for _ in range(image_each):
                x, y = get_random_crop_coords(
                    x_width=x_width,
                    x_height=x_height,
                    crop_w=cfg.img_w,
                    crop_h=cfg.img_h,
                    split_ratio=cfg.split_ratio,
                    subset=subset
                )
                crop_coords.append((x, y, False))

        for crop_x, crop_y, need_pad in crop_coords:
            if need_pad:
                src_roi = crop_with_padding(src_img, crop_x, crop_y, cfg.img_w, cfg.img_h)
                label_roi = crop_with_padding(label_img, crop_x, crop_y, cfg.img_w, cfg.img_h)
                building_roi = crop_with_padding(label_building, crop_x, crop_y, cfg.img_w, cfg.img_h)
            else:
                src_roi = src_img[
                    crop_y:crop_y + cfg.img_h,
                    crop_x:crop_x + cfg.img_w,
                    :
                ]
                label_roi = label_img[
                    crop_y:crop_y + cfg.img_h,
                    crop_x:crop_x + cfg.img_w,
                    :
                ]
                building_roi = label_building[
                    crop_y:crop_y + cfg.img_h,
                    crop_x:crop_x + cfg.img_w
                ]

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)

            save_patch_results(
                g_count=g_count,
                src_roi=src_roi,
                label_roi=label_roi,
                building_roi=building_roi,
                output_root=cfg.output_root
            )

            g_count += 1

            if cfg.verbose and g_count % 100 == 0:
                print(f'saved patch count = {g_count}')

    print(f'Finished! Total patches saved: {g_count}')

    # sliding 模式下，生成可复现的 train/val/test 划分
    if cfg.crop_mode == 'sliding' and subset == 'all':
        # 从 cfg 读取划分比例（通过 args 传入）
        train_ratio = getattr(cfg, 'train_ratio', 0.7)
        val_ratio = getattr(cfg, 'val_ratio', 0.1)
        random_split_indices(g_count, cfg.output_root, train_ratio=train_ratio, val_ratio=val_ratio)


# =========================
# 7. 使用示例
# =========================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成训练数据集')
    parser.add_argument('--input', '-i', required=True, help='输入数据根目录')
    parser.add_argument('--output', '-o', default=None, help='输出目录')
    parser.add_argument('--patch_num', '-n', type=int, default=100, help='每张图采样patch数')
    parser.add_argument('--patch_size', '-s', type=int, default=256, help='patch大小')
    parser.add_argument('--split_ratio', '-r', type=float, default=7/8, help='训练集比例')
    parser.add_argument('--subset', '-u', default='all', choices=['train', 'val', 'all'], help='生成哪个集合')
    parser.add_argument('--mode', '-m', default='random', choices=['random', 'sliding'], help='裁剪模式')
    parser.add_argument('--stride', type=int, default=256, help='滑动窗口步长')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='训练集比例 (sliding模式有效)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例 (sliding模式有效)')
    args = parser.parse_args()

    # 设置输出目录
    output_root = args.output if args.output else args.input + '_patches'

    # 获取图像文件名列表
    image_sets = sorted(os.listdir(os.path.join(args.input, 'src')))
    print(f"找到 {len(image_sets)} 个图像文件")

    cfg = DatasetConfig(
        data_root=args.input,
        output_root=output_root,
        img_h=args.patch_size,
        img_w=args.patch_size,
        stride_h=args.stride,
        stride_w=args.stride,
        image_num=args.patch_num,
        crop_mode=args.mode,
        split_ratio=args.split_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=42,
        verbose=True
    )

    if args.subset == 'all':
        if cfg.crop_mode == 'sliding':
            # sliding 模式：生成所有 patches，然后按索引随机划分
            print("\n=== 生成所有 patches (sliding模式) ===")
            create_dataset(image_sets=image_sets, cfg=cfg, mode='normal', subset='all')
            # random_split_indices 已在 create_dataset 内部调用
        else:
            # random 模式：按空间区域划分
            print("\n=== 生成训练集 ===")
            create_dataset(image_sets=image_sets, cfg=cfg, mode='normal', subset='train')
            print("\n=== 生成验证集 ===")
            create_dataset(image_sets=image_sets, cfg=cfg, mode='normal', subset='val')
    else:
        create_dataset(image_sets=image_sets, cfg=cfg, mode='normal', subset=args.subset)

    print(f"\n完成! 输出目录: {output_root}")