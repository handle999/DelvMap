"""DSFNet 预测掩膜 → AdaMap 可用的 shapefile 路网

流程：
    1. 读 results/{name}/all_{epoch}/images_full/{idx}_pred_traj_img.png
    2. 拼回全图 5625×6610 (max-overlap)
    3. 二值化 + 形态学 closing + 去小连通块 + 骨架化 (skimage)
    4. 骨架像素 → networkx graph (8-邻接 BFS，degree=1/≥3 为节点)
    5. 像素 → Web Mercator (EPSG:3857) → WGS84
    6. 写 shapefile (复用 tptk.common.road_network.store_rn_shp)

依赖：
    pip install scikit-image opencv-python tqdm
    （tptk 自带依赖：gdal, rtree, networkx<3）

用法：
    python adaptive_map_completion/raster_to_shp.py \
        --pred_dir results/delvmap_exp2/all_357/images_full \
        --out_dir  inferred_rn_xa \
        [--head pred_traj_img | pred_src_traj_img] \
        [--bin_thresh 128] [--min_obj 50] [--simplify_eps 2.0] \
        [--save_full_pred]
"""
import os
import sys
import math
import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
import networkx as nx
from rtree import Rtree
from tqdm import tqdm

# tptk 是 adaptive_map_completion 内部包；按相对路径加载
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.insert(0, _THIS_DIR)   # 让 `from tptk...` 与 walkway_completion 保持一致
sys.path.insert(0, _REPO_ROOT)  # for dataset.patch_index_map

from tptk.common.spatial_func import SPoint, distance  # noqa: E402
from tptk.common.road_network import UndirRoadNetwork, store_rn_shp  # noqa: E402
from tptk.common.mbr import MBR  # noqa: E402
from tptk.common.douglas_peucker import DouglasPeucker  # noqa: E402

from dataset.patch_index_map import PatchGridConfig, list_coords  # noqa: E402


# ============================================================
# CONFIG (与 dataset/prepare_dataset.py:27-37 一致)
# ============================================================
GEO_CONFIG = {
    'lat_min': 34.206385,
    'lat_max': 34.279658,
    'lon_min': 108.917423,
    'lon_max': 108.99286,
    'img_w': 5625,
    'img_h': 6610,
}


# ============================================================
# 1. WGS84 ↔ Web Mercator ↔ pixel
# ============================================================
def wgs84_to_mercator(lon, lat):
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


def mercator_to_wgs84(x_m, y_m):
    lon = x_m / 20037508.34 * 180.0
    lat = math.atan(math.exp(y_m / 20037508.34 * math.pi)) * 360.0 / math.pi - 90.0
    return lon, lat


def _mercator_bbox(cfg):
    x_min, y_min = wgs84_to_mercator(cfg['lon_min'], cfg['lat_min'])
    x_max, y_max = wgs84_to_mercator(cfg['lon_max'], cfg['lat_max'])
    return x_min, y_min, x_max, y_max


def pixel_to_geo(px, py, cfg=GEO_CONFIG):
    """像素坐标 (px, py) → (lat, lon)。py 自顶向下，与 OpenCV 一致。"""
    x_min, y_min, x_max, y_max = _mercator_bbox(cfg)
    x_m = px / cfg['img_w'] * (x_max - x_min) + x_min
    y_m = y_max - py / cfg['img_h'] * (y_max - y_min)
    lon, lat = mercator_to_wgs84(x_m, y_m)
    return lat, lon


def geo_to_pixel(lat, lon, cfg=GEO_CONFIG):
    """与 dataset/prepare_dataset.py:69-84 完全一致，做自洽测试用。"""
    x_min, y_min, x_max, y_max = _mercator_bbox(cfg)
    x_m, y_m = wgs84_to_mercator(lon, lat)
    px = (x_m - x_min) / (x_max - x_min) * cfg['img_w']
    py = (y_max - y_m) / (y_max - y_min) * cfg['img_h']
    return px, py


def _self_test_coord():
    """像素 ↔ 经纬度往返误差应 < 1e-9"""
    samples = [
        (0, 0),
        (GEO_CONFIG['img_w'] - 1, GEO_CONFIG['img_h'] - 1),
        (1234.5, 5678.5),
    ]
    for px, py in samples:
        lat, lon = pixel_to_geo(px, py)
        px2, py2 = geo_to_pixel(lat, lon)
        err = max(abs(px - px2), abs(py - py2))
        assert err < 1e-6, f"coord roundtrip err {err} at ({px},{py})"
    print("[self-test] pixel↔geo roundtrip OK")


# ============================================================
# 2. 拼图：max-overlap
# ============================================================
def stitch_full_pred(pred_dir, head='pred_traj_img', cfg=PatchGridConfig()):
    """读 {idx}_{head}.png 并 max 拼回 cfg.img_h x cfg.img_w 灰度大图。"""
    full = np.zeros((cfg.img_h, cfg.img_w), dtype=np.uint8)
    coords = list_coords(cfg)
    n_total = len(coords)
    n_loaded = 0
    n_missing = 0
    for idx, (cx, cy) in enumerate(tqdm(coords, desc='stitch')):
        path = os.path.join(pred_dir, f'{idx}_{head}.png')
        if not os.path.exists(path):
            n_missing += 1
            continue
        patch = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if patch is None:
            n_missing += 1
            continue
        h_eff = min(cfg.patch_h, cfg.img_h - cy)
        w_eff = min(cfg.patch_w, cfg.img_w - cx)
        region = full[cy:cy + h_eff, cx:cx + w_eff]
        np.maximum(region, patch[:h_eff, :w_eff], out=region)
        n_loaded += 1
    print(f"[stitch] {n_loaded}/{n_total} patches loaded, {n_missing} missing.")
    if n_missing > 0:
        print(f"[stitch][warn] {n_missing} patches missing; full_pred has gaps.")
    return full


# ============================================================
# 3. 二值化 + 形态学 + 骨架化
# ============================================================
def binarize_and_skeletonize(full_pred, bin_thresh=128, close_ksize=3, min_obj=50):
    """
    full_pred: HxW uint8 (0 / 255 灰度)
    返回 (binary uint8 0/1, skeleton bool)
    """
    from skimage.morphology import remove_small_objects, skeletonize

    binary = (full_pred > bin_thresh).astype(np.uint8)
    if close_ksize and close_ksize > 1:
        kernel = np.ones((close_ksize, close_ksize), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if min_obj and min_obj > 0:
        binary_bool = binary.astype(bool)
        binary_bool = remove_small_objects(binary_bool, min_size=min_obj)
        binary = binary_bool.astype(np.uint8)
    skeleton = skeletonize(binary.astype(bool))
    print(f"[binarize] road pixels: {int(binary.sum())}, skeleton pixels: {int(skeleton.sum())}")
    return binary, skeleton


# ============================================================
# 4. 骨架 → graph (networkx 像素图 + degree-2 链合并，可靠处理交叉/环)
# ============================================================
# 用 8-邻接建图，但对角边加保护：仅当对角相邻两像素没有共同的正交邻居时才连。
# 纯 4-邻接会切断 45° 斜路的对角像素过渡，把一条连续斜路碎成多段
# (骨架的 4-连通分量数远大于实际路数)；纯 8-邻接又会在十字/T字交叉处把
# 正交线的相邻像素通过对角连成虚假高 degree 节点。带保护的 8-邻接兼顾两者：
# 斜路对角过渡 (无共同正交邻居) 正确连通，交叉处擦肩 (有共同正交邻居) 不误连。
_NB4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NB_DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbors4(skel, y, x):
    H, W = skel.shape
    for dy, dx in _NB4:
        ny, nx_ = y + dy, x + dx
        if 0 <= ny < H and 0 <= nx_ < W and skel[ny, nx_]:
            yield ny, nx_


def _build_pixel_graph(skel_bool):
    """每个骨架像素一个节点，带对角保护的 8-邻接建无向图。

    正交方向 (4-邻接) 无条件连；对角方向仅当两端无共同正交邻居时连
    (有共同正交邻居 = 交叉处擦肩, 不连, 避免虚假拓扑)。
    """
    H, W = skel_bool.shape
    g = nx.Graph()
    ys, xs = np.where(skel_bool)
    g.add_nodes_from(zip(ys.tolist(), xs.tolist()))
    skel = skel_bool  # bool 数组, 索引快
    for y, x in zip(ys, xs):
        # 正交邻居
        ortho_nbrs = []
        for dy, dx in _NB4:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W and skel[ny, nx_]:
                g.add_edge((y, x), (ny, nx_))
                ortho_nbrs.append((ny, nx_))
        # 对角邻居: 仅当无共同正交邻居时连
        ortho_set = set(ortho_nbrs)
        for dy, dx in _NB_DIAG:
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W and skel[ny, nx_]:
                # 共同正交邻居 = (ny,x) 和 (y,nx_) 这两个正交位
                if (ny, x) in ortho_set or (y, nx_) in ortho_set:
                    continue  # 交叉处擦肩, 不连对角
                g.add_edge((y, x), (ny, nx_))
    return g


def skeleton_to_graph(skel):
    """
    骨架 (HxW bool) → list of edges，每条边是像素序列 [(y, x), ...]。
    图节点 = 端点 (deg=1) 或 交叉点 (deg>=3)；deg=2 的中间像素被吸收进边。
    纯环路 (整连通块 deg 全为 2) 会被断成一条闭合边。

    实现：先建带对角保护的 8-邻接像素 networkx 图 (_build_pixel_graph)，
    再沿 degree-2 链追踪。8-邻接保证斜路对角过渡连通，对角保护避免交叉处
    虚假拓扑。networkx 保证度数/连通性计算正确，方环不会碎裂。
    相比手写 BFS，networkx 保证度数/连通性计算正确，交叉点 (deg=4) 不会
    产生虚假短边，方环不会碎裂。
    """
    skel_bool = skel.astype(bool)
    pg = _build_pixel_graph(skel_bool)

    # 1. 图节点 = deg != 2 的骨架像素 (端点 deg=1 / 交叉点 deg>=3)
    is_node = {p for p, d in pg.degree() if d != 2}
    # 孤立像素 (deg=0) 丢弃，不产生边
    is_node = {p for p in is_node if pg.degree(p) >= 1}

    edges = []
    walked = set()  # 已被某条边吸收的 deg=2 中间像素

    # 2. 从每个图节点出发，沿 deg=2 链追踪到下一个图节点
    for start in tqdm(sorted(is_node), desc='trace edges'):
        for nbr in sorted(pg.neighbors(start)):
            # 起点邻居本身就是图节点 → length-2 的短边 (start, nbr)
            if nbr in is_node:
                ekey = frozenset((start, nbr))
                if ekey not in walked:
                    edges.append([start, nbr])
                    # node-node 短边不占 deg=2 像素，用 frozenset 去重即可
                    walked.add(ekey)
                continue
            # 否则 nbr 是 deg=2 链的入口，沿链走到下一个图节点
            if nbr in walked:
                continue
            chain = [start, nbr]
            prev, cur = start, nbr
            while cur not in is_node:
                # deg=2 像素恰好有两个邻居，取“不是 prev”的那个
                nxts = [n for n in pg.neighbors(cur) if n != prev]
                if not nxts:
                    break  # 悬空 (理论上 deg=2 不会悬空，防御性处理)
                nxt = nxts[0]
                chain.append(nxt)
                prev, cur = cur, nxt
            # 标记链上所有 deg=2 中间像素已吸收 (不含两端图节点)
            for p in chain[1:-1]:
                walked.add(p)
            edges.append(chain)

    # 3. 纯环路：整连通块 deg 全为 2，没有任何图节点，上面会漏掉。
    #    从未被吸收的像素出发，沿环走一圈。
    covered = set()
    for e in edges:
        covered.update(e)
    for p in pg.nodes():
        if p in covered:
            continue
        # 找到这个纯环所在的连通块，任取起点断开成一条闭合边
        ring = [p]
        covered.add(p)
        prev, cur = None, p
        while True:
            nxts = [n for n in pg.neighbors(cur) if n != prev and n not in covered]
            if not nxts:
                break
            nxt = nxts[0]
            ring.append(nxt)
            covered.add(nxt)
            prev, cur = cur, nxt
        if len(ring) >= 3:
            ring.append(ring[0])  # 闭合
            edges.append(ring)

    print(f"[graph] pixel nodes: {pg.number_of_nodes()}, "
          f"graph nodes (deg!=2): {len(is_node)}, edges: {len(edges)}")
    return edges


# ============================================================
# 5. pixel-edges → networkx UndirRoadNetwork (WGS84)
# ============================================================
def pixel_edges_to_rn(edges, simplify_eps_pixel=2.0, geo_cfg=GEO_CONFIG):
    """
    edges: list of [(y, x), ...]  像素序列
    simplify_eps_pixel: Douglas-Peucker 简化阈值 (像素)
    返回: UndirRoadNetwork
    """
    g = nx.Graph()
    edge_idx = {}
    edge_spatial_idx = Rtree()
    avail_eid = 0

    # 用一个简单像素层面的 DP 简化（基于点到直线的欧氏距离）
    def _dp_pixel(pts, eps):
        if len(pts) <= 2:
            return pts
        # iterative DP
        keep = [False] * len(pts)
        keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            i, j = stack.pop()
            if j - i < 2:
                continue
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx, dy = x2 - x1, y2 - y1
            denom = math.hypot(dx, dy) or 1.0
            max_d, max_k = 0.0, -1
            for k in range(i + 1, j):
                x0, y0 = pts[k]
                d = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / denom
                if d > max_d:
                    max_d, max_k = d, k
            if max_d > eps:
                keep[max_k] = True
                stack.append((i, max_k))
                stack.append((max_k, j))
        return [pts[k] for k in range(len(pts)) if keep[k]]

    skipped_short = 0
    skipped_loop = 0
    for raw_edge in tqdm(edges, desc='to-shp'):
        # raw_edge: [(y,x), ...] -> 转成 (x, y) 做 DP
        pts_xy = [(x, y) for (y, x) in raw_edge]
        pts_xy = _dp_pixel(pts_xy, simplify_eps_pixel)
        if len(pts_xy) < 2:
            skipped_short += 1
            continue

        # 像素 → WGS84，节点 key 用 (lng, lat) 与 AdaMap 内部约定一致
        coords_sp = []
        node_keys = []
        for (px, py) in pts_xy:
            lat, lon = pixel_to_geo(px, py)
            coords_sp.append(SPoint(lat, lon))
            node_keys.append((lon, lat))

        u, v = node_keys[0], node_keys[-1]
        if u == v and len(node_keys) <= 2:
            skipped_loop += 1
            continue

        # 已存在则跳过（避免无向图重边）
        if g.has_edge(u, v):
            continue

        edge_data = {
            'eid': avail_eid,
            'coords': coords_sp,
            'length': sum(distance(coords_sp[i], coords_sp[i + 1]) for i in range(len(coords_sp) - 1)),
        }
        g.add_node(u, pt=SPoint(u[1], u[0]))
        g.add_node(v, pt=SPoint(v[1], v[0]))
        g.add_edge(u, v, **edge_data)
        mbr = MBR.cal_mbr(coords_sp)
        edge_spatial_idx.insert(avail_eid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        edge_idx[avail_eid] = (u, v)
        avail_eid += 1

    print(f"[rn] kept edges: {avail_eid} | skipped short: {skipped_short} | skipped pure-loop: {skipped_loop}")
    print(f"[rn] # nodes: {g.number_of_nodes()}, # edges: {g.number_of_edges()}")
    return UndirRoadNetwork(g, edge_spatial_idx, edge_idx)


# ============================================================
# 6. 主流程
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', required=True, help='infer_all 输出目录: results/.../all_*/images_full')
    p.add_argument('--out_dir', required=True, help='shapefile 输出目录 (会写 edges.shp / nodes.shp)')
    p.add_argument('--head', default='pred_traj_img',
                   choices=['pred_traj_img', 'pred_src_traj_img'],
                   help='用哪个预测头作为掩膜')
    p.add_argument('--bin_thresh', type=int, default=128)
    p.add_argument('--close_ksize', type=int, default=3, help='形态学闭运算 kernel 大小 (1=禁用)')
    p.add_argument('--min_obj', type=int, default=50, help='移除小连通块阈值 (像素)')
    p.add_argument('--simplify_eps', type=float, default=2.0, help='Douglas-Peucker 像素阈值')
    p.add_argument('--save_full_pred', action='store_true', help='保存拼出来的全图灰度 PNG，便于人工核对')
    args = p.parse_args()

    # 0. 自洽测试
    _self_test_coord()

    # 1. 拼图
    print(f"\n[1/4] Stitching patches from {args.pred_dir} (head={args.head}) ...")
    full_pred = stitch_full_pred(args.pred_dir, head=args.head)

    if args.save_full_pred:
        os.makedirs(args.out_dir, exist_ok=True)
        full_path = os.path.join(args.out_dir, f'full_pred_{args.head}.png')
        cv2.imwrite(full_path, full_pred)
        print(f"  full_pred saved to {full_path}")

    # 2. 二值化 + 骨架化
    print(f"\n[2/4] Binarize + skeletonize ...")
    binary, skeleton = binarize_and_skeletonize(
        full_pred,
        bin_thresh=args.bin_thresh,
        close_ksize=args.close_ksize,
        min_obj=args.min_obj,
    )
    if args.save_full_pred:
        cv2.imwrite(os.path.join(args.out_dir, 'binary.png'), binary * 255)
        cv2.imwrite(os.path.join(args.out_dir, 'skeleton.png'), skeleton.astype(np.uint8) * 255)

    # 3. 骨架 → graph
    print(f"\n[3/4] Skeleton → graph ...")
    edges = skeleton_to_graph(skeleton)
    if not edges:
        print("[error] no edges extracted; check binarize/skeletonize params.")
        return

    # 4. 像素 graph → WGS84 UndirRoadNetwork → shp
    print(f"\n[4/4] Pixel graph → WGS84 → shapefile ...")
    rn = pixel_edges_to_rn(edges, simplify_eps_pixel=args.simplify_eps)

    os.makedirs(args.out_dir, exist_ok=True)
    # store_rn_shp 会在 out_dir 下生成 edges.shp / nodes.shp
    # 但实测 nx.write_shp 要求目录非空时按 dataset 写入；保险起见传 out_dir 本身
    print(f"\nWriting shp to {args.out_dir} ...")
    store_rn_shp(rn, args.out_dir)
    print(f"\n[done] inferred_rn → {args.out_dir}")


if __name__ == '__main__':
    main()
