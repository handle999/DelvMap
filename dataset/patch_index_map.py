"""patch idx ↔ (crop_x, crop_y) 映射

复刻 dataset/create_dataset.py 中 get_sliding_window_coords 的滑窗逻辑（padding=True），
保证与训练数据 patch 化时使用的索引完全一致：
    while y < img_h:
        while x < img_w:
            yield (x, y); x += stride
        y += stride

默认配置（与生成 dataset/xian_2019_delvmap/ 时保持一致）：
    img_w=5625, img_h=6610, patch=256, stride=128 → 共 2288 个 patch (52 行 × 44 列)

只读工具，不修改任何外部文件。
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class PatchGridConfig:
    img_w: int = 5625
    img_h: int = 6610
    patch_w: int = 256
    patch_h: int = 256
    stride_w: int = 128
    stride_h: int = 128


def list_coords(cfg: PatchGridConfig = PatchGridConfig()) -> List[Tuple[int, int]]:
    """按 create_dataset.py 的 padding=True 模式枚举所有 patch 左上角坐标。"""
    coords = []
    y = 0
    while y < cfg.img_h:
        x = 0
        while x < cfg.img_w:
            # padding=True 时边缘 patch 也被保留（need_pad 不影响坐标本身）
            coords.append((x, y))
            x += cfg.stride_w
        y += cfg.stride_h
    return coords


def idx_to_xy(idx: int, cfg: PatchGridConfig = PatchGridConfig()) -> Tuple[int, int]:
    coords = list_coords(cfg)
    return coords[idx]


def total_patches(cfg: PatchGridConfig = PatchGridConfig()) -> int:
    return len(list_coords(cfg))


if __name__ == '__main__':
    cfg = PatchGridConfig()
    coords = list_coords(cfg)
    print(f"img: {cfg.img_w}x{cfg.img_h}, patch: {cfg.patch_w}, stride: {cfg.stride_w}")
    print(f"total patches: {len(coords)}")
    print(f"first: {coords[0]}, last: {coords[-1]}")
    # sanity: max(x)+patch_w 应 ≥ img_w，max(y)+patch_h 应 ≥ img_h（覆盖完整）
    max_x = max(x for x, _ in coords)
    max_y = max(y for _, y in coords)
    print(f"max_x={max_x} (+patch={max_x + cfg.patch_w} vs img_w={cfg.img_w})")
    print(f"max_y={max_y} (+patch={max_y + cfg.patch_h} vs img_h={cfg.img_h})")
