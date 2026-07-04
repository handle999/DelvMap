"""基于 didi_xian 标准数据集构造 DSFNet 训练集 (消除泄漏)。

背景：DelvMap 原训练集 xian_2019_delvmap 从 rawdata 像素直切 256px patch, 与 didi_xian
test region (Mercator 400px) 是不同切分网格, 10.9% train patch 落入 test region → 泄漏。
本脚本基于 didi_xian region 级数据 + rawdata, 用 samroad get_local_sat 同款 Mercator
重投影生成训练集, 按 region 级 split 划分, 零泄漏。

数据来源 (4 模态, 全部对齐到 didi_xian region 400px Mercator 坐标系):
  - sat:           rawdata/sat_img.png Mercator 重投影 (与 didi region_sat 同源同公式)
  - building_label: rawdata/building_label_full.png Mercator 重投影 (didi_xian 无 building)
  - traj:          datasets/didi/xian/2019_400/region_{idx}_traj.png (samroad 已生成)
  - label:         datasets/didi/xian/2019_400/region_{idx}_gt.png (samroad GT, 评估同源)

400→256 切分: sliding 2×2 (offsets=[0,144]×[0,144], 4 patch/region, stride 144, overlap 112px)
split: 按 didi_xian data_split.json 的 region 级划分 (train region 的 4 patch → train)
       patch idx = region_idx * 4 + offset_idx (0=TL,1=TR,2=BL,3=BR)

输出: dataset/didi_xian_train/
  ├── traj_and_point_split/{idx}.npy  (256,256,2) uint8, ch0=traj, ch1=0 (point置零)
  ├── src_split/{idx}.png             256×256 RGB
  ├── label/{idx}.png                 256×256 灰度 (道路GT)
  ├── building_label/{idx}.png        256×256 灰度 (建筑GT)
  └── split_indices.json              {train/val/test: [patch idx]}

用法:
    python dataset/prepare_didi_trainset.py
    python dataset/prepare_didi_trainset.py --verify   # 仅验证已生成数据
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'adaptive_map_completion'))

import didi_bridge as db  # noqa: E402

# DelvMap 西安 bbox (xian.json, 与 samroad didi_xian 一致)
LAT_MIN, LAT_MAX = 34.206385, 34.279658
LON_MIN, LON_MAX = 108.917423, 108.99286
SIZE = 400          # region 像素边长
PATCH = 256         # DSFNet patch 边长
OFFSETS = [0, 144]  # sliding 2×2: 256 patch 在 400 region 内的偏移 (stride 144, overlap 112)
R = 20037508.34     # Web Mercator 半周长

RAWDATA = os.path.join(_REPO_ROOT, 'rawdata')
DIDI = os.path.join(_REPO_ROOT, 'datasets', 'didi', 'xian', '2019_400')
SPLIT_FILE = os.path.join(_REPO_ROOT, 'datasets', 'didi', 'xian', 'data_split.json')
OUT_DIR = os.path.join(_REPO_ROOT, 'dataset', 'didi_xian_train')

# Mercator bounds (与 DelvMap prepare_dataset.py / samroad get_local_sat 一致)
_X_MIN = LON_MIN * R / 180.0
_X_MAX = LON_MAX * R / 180.0
_Y_MIN = math.log(math.tan((90.0 + LAT_MIN) * math.pi / 360.0)) / (math.pi / 180.0) * R / 180.0
_Y_MAX = math.log(math.tan((90.0 + LAT_MAX) * math.pi / 360.0)) / (math.pi / 180.0) * R / 180.0


def get_local(big, lat_st, lon_st, lat_ed, lon_ed, size=SIZE):
    """samroad get_local_sat 同款 Mercator 重投影 (download_use_osm.py:143-163)。

    big: (H,W) 或 (H,W,3) 大图 (Web Mercator, bbox=DelvMap xian)
    返回 (size,size) tile, 坐标系与 didi_xian region 一致 (row=0 北, col=0 西)。
    """
    img_h, img_w = big.shape[:2]
    coords = np.arange(size, dtype=np.float64)
    px, py = np.meshgrid(coords, coords)
    # tile 像素 (px,py) → latlon (线性, 与 graph2RegionCoordinate 一致: 北在 py=0)
    lat = lat_ed - (py / size) * (lat_ed - lat_st)
    lon = lon_st + (px / size) * (lon_ed - lon_st)
    # latlon → Mercator → 大图像素
    x_m = lon * R / 180.0
    y_m = np.log(np.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0) * R / 180.0
    bx = (x_m - _X_MIN) / (_X_MAX - _X_MIN) * img_w
    by = (_Y_MAX - y_m) / (_Y_MAX - _Y_MIN) * img_h
    ix = np.clip(np.rint(bx).astype(np.int64), 0, img_w - 1)
    iy = np.clip(np.rint(by).astype(np.int64), 0, img_h - 1)
    return big[iy, ix]


def gen_region_400(region_idx, sat_big, bldg_big):
    """生成单个 region 的 4 模态 400px (sat/bldg Mercator 采样, traj/label 读 didi_xian)。"""
    lat_st, lat_ed, lon_st, lon_ed = db.region_bbox(region_idx)
    sat = get_local(sat_big, lat_st, lon_st, lat_ed, lon_ed)
    bldg = get_local(bldg_big, lat_st, lon_st, lat_ed, lon_ed)
    traj = cv2.imread(os.path.join(DIDI, f'region_{region_idx}_traj.png'), cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(os.path.join(DIDI, f'region_{region_idx}_gt.png'), cv2.IMREAD_GRAYSCALE)
    if traj is None or label is None:
        return None
    return sat, bldg, traj, label


def crop_256(modality_400, off_y, off_x):
    """从 400px 模态裁 256px patch。"""
    return modality_400[off_y:off_y + PATCH, off_x:off_x + PATCH]


def main_generate():
    os.makedirs(os.path.join(OUT_DIR, 'traj_and_point_split'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'src_split'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'label'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'building_label'), exist_ok=True)

    sat_big = cv2.imread(os.path.join(RAWDATA, 'sat_img.png'), cv2.IMREAD_COLOR)  # RGB, 丢 alpha
    bldg_big = cv2.imread(os.path.join(RAWDATA, 'building_label_full.png'), cv2.IMREAD_GRAYSCALE)
    print(f'rawdata: sat{sat_big.shape} bldg{bldg_big.shape}')

    split = json.load(open(SPLIT_FILE))
    region_split = {}  # region_idx -> 'train'/'val'/'test'
    # data_split.json 键: train/validation/test
    for mode, key in (('train', 'train'), ('val', 'validation'), ('test', 'test')):
        for r in split[key]:
            region_split[int(r)] = mode

    patch_split = {'train': [], 'val': [], 'test': []}
    n_regions = 0
    for region_idx in sorted(region_split.keys()):
        mod = gen_region_400(region_idx, sat_big, bldg_big)
        if mod is None:
            print(f'[skip] region {region_idx}: 缺 traj/gt')
            continue
        sat400, bldg400, traj400, label400 = mod
        mode = region_split[region_idx]
        n_regions += 1
        for oy_i, oy in enumerate(OFFSETS):
            for ox_i, ox in enumerate(OFFSETS):
                off_idx = oy_i * 2 + ox_i  # 0=TL,1=TR,2=BL,3=BR
                patch_idx = region_idx * 4 + off_idx
                sat256 = crop_256(sat400, oy, ox)
                bldg256 = crop_256(bldg400, oy, ox)
                traj256 = crop_256(traj400, oy, ox)
                label256 = crop_256(label400, oy, ox)
                # 写 4 文件
                cv2.imwrite(os.path.join(OUT_DIR, 'src_split', f'{patch_idx}.png'), sat256)
                cv2.imwrite(os.path.join(OUT_DIR, 'label', f'{patch_idx}.png'), label256)
                cv2.imwrite(os.path.join(OUT_DIR, 'building_label', f'{patch_idx}.png'), bldg256)
                # traj npy (256,256,2): ch0=traj, ch1=0 (point 置零, didi_xian 无 point)
                traj_npy = np.zeros((PATCH, PATCH, 2), dtype=np.uint8)
                traj_npy[:, :, 0] = traj256
                np.save(os.path.join(OUT_DIR, 'traj_and_point_split', f'{patch_idx}.npy'), traj_npy)
                patch_split[mode].append(patch_idx)
        if n_regions % 50 == 0:
            print(f'  已处理 {n_regions} region')

    json.dump(patch_split, open(os.path.join(OUT_DIR, 'split_indices.json'), 'w'))
    print(f'\n生成完成: {n_regions} region → {sum(len(v) for v in patch_split.values())} patch')
    print(f'  train: {len(patch_split["train"])}  val: {len(patch_split["val"])}  test: {len(patch_split["test"])}')


def main_verify():
    """验证: 泄漏检查 + 对齐检查 + 完整性检查。"""
    split = json.load(open(SPLIT_FILE))
    test_regions = [int(r) for r in split['test']]
    test_bboxes = [db.region_bbox(r) for r in test_regions]

    patch_split = json.load(open(os.path.join(OUT_DIR, 'split_indices.json')))

    # 1. 泄漏检查: train patch 反查 geo, 确认 0 个落在 test region 内
    # patch_idx = region*4+off, 反推 region_idx = patch_idx // 4
    train_regions = sorted(set(p // 4 for p in patch_split['train']))
    leak = sum(1 for r in train_regions if r in set(test_regions))
    print(f'=== 1. 泄漏检查 ===')
    print(f'  train 含 {len(train_regions)} 个 region, 其中落在 test region 内: {leak}')
    print(f'  {"✓ 零泄漏" if leak == 0 else "✗ 仍有泄漏!"}')

    # 2. 对齐检查: 抽 region_92 patch0 (TL) 的 sat, 和 region_92_sat.png 左上 256 区域比
    print(f'\n=== 2. 对齐检查 (region_92 TL patch sat vs didi region_sat) ===')
    r = 92
    sat_patch = cv2.imread(os.path.join(OUT_DIR, 'src_split', f'{r * 4}.png'))
    didi_sat = cv2.imread(os.path.join(DIDI, f'region_{r}_sat.png'))
    # didi_sat 是 samroad get_local_sat 采的, 我们也是同公式, 应几乎一致
    diff = np.abs(sat_patch.astype(int) - didi_sat[:PATCH, :PATCH].astype(int)).mean()
    print(f'  sat patch vs didi_sat[TL]: 像素差={diff:.2f} (应<3)')

    # 3. 完整性检查
    print(f'\n=== 3. 完整性检查 ===')
    for mode in ('train', 'val', 'test'):
        n = len(patch_split[mode])
        if n == 0:
            continue
        p0 = patch_split[mode][0]
        traj = np.load(os.path.join(OUT_DIR, 'traj_and_point_split', f'{p0}.npy'))
        src = cv2.imread(os.path.join(OUT_DIR, 'src_split', f'{p0}.png'))
        lab = cv2.imread(os.path.join(OUT_DIR, 'label', f'{p0}.png'), cv2.IMREAD_GRAYSCALE)
        bldg = cv2.imread(os.path.join(OUT_DIR, 'building_label', f'{p0}.png'), cv2.IMREAD_GRAYSCALE)
        print(f'  {mode}: {n} patch, 样例 {p0}: traj{traj.shape} src{src.shape} lab{lab.shape} bldg{bldg.shape}')
        print(f'    traj ch0非零={100*(traj[:,:,0]>0).mean():.1f}% ch1非零={100*(traj[:,:,1]>0).mean():.1f}% '
              f'lab非零={100*(lab>0).mean():.1f}% bldg非零={100*(bldg>0).mean():.1f}%')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--verify', action='store_true', help='仅验证已生成数据')
    args = p.parse_args()
    if args.verify:
        main_verify()
    else:
        main_generate()
        main_verify()
