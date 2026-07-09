"""把 xian_2019_delvmap/building_label/ 的 2288 个 256px patch 拼回 5625×6610 全图。

prepare_didi_trainset.py 需要 rawdata/building_label_full.png (建筑GT全图),
didi_xian 无 building 数据, 从 DelvMap 256patch 拼回再 Mercator 重投影采样。

拼回方式: max-overlap (和 raster_to_shp.stitch_full_pred 一致),
patch 256 stride 128, 5625×6610 大图, 边缘 patch clamp。

用法:
    python dataset/stitch_building_label_full.py
    # 输入: dataset/xian_2019_delvmap/building_label/{idx}.png
    # 输出: rawdata/building_label_full.png

    # 可选拼回 road label (验证用)
    python dataset/stitch_building_label_full.py --src label --out rawdata/label_full.png
"""
import os
import sys
import argparse

import numpy as np
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'adaptive_map_completion'))

from dataset.patch_index_map import PatchGridConfig, list_coords  # noqa: E402


def stitch(src_subdir, out_path, cfg=PatchGridConfig()):
    """从 {xian_2019_delvmap}/{src_subdir}/{idx}.png max-overlap 拼回全图。"""
    data_root = os.path.join(_REPO_ROOT, 'dataset', 'xian_2019_delvmap')
    coords = list_coords(cfg)
    full = np.zeros((cfg.img_h, cfg.img_w), dtype=np.uint8)
    n_loaded = 0
    for idx, (cx, cy) in enumerate(coords):
        path = os.path.join(data_root, src_subdir, f'{idx}.png')
        if not os.path.exists(path):
            continue
        patch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if patch is None:
            continue
        h_eff = min(cfg.patch_h, cfg.img_h - cy)
        w_eff = min(cfg.patch_w, cfg.img_w - cx)
        region = full[cy:cy + h_eff, cx:cx + w_eff]
        np.maximum(region, patch[:h_eff, :w_eff], out=region)
        n_loaded += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, full)
    print(f'[stitch] {src_subdir}: {n_loaded}/{len(coords)} patch -> {full.shape} '
          f'非零={100 * (full > 0).mean():.1f}% -> {out_path}')
    return full


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src', default='building_label',
                   help='xian_2019_delvmap 下的子目录名 (building_label/label)')
    p.add_argument('--out', default=None, help='输出路径 (默认 rawdata/{src}_full.png)')
    args = p.parse_args()
    out = args.out or os.path.join(_REPO_ROOT, 'rawdata', f'{args.src}_full.png')
    stitch(args.src, out)


if __name__ == '__main__':
    main()
