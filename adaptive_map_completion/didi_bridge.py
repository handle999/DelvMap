"""didi_xian ↔ DelvMap AMC 中间件。

把 samroad didi_xian 的 partial/gt/active pickle 与 DelvMap DSFNet 输出对接到 AMC，
再把 AMC 输出转回 didi_xian 邻接字典 pickle 供 samroad 原生 APLS/TOPO 评估。

坐标系：统一用 didi_xian 的线性 lat/lon ↔ (row,col) 映射（与 download_use_osm.py 一致）。
  - region (row,col) ↔ WGS84 (lat,lon): lat = lat_ed - (row/size)*(lat_ed-lat_st)
                                      lon = lon_st + (col/size)*(lon_ed-lon_st)
  - row=0 在北 (lat_ed), col=0 在西 (lon_st); size=400
  - DelvMap 整图像素 ↔ WGS84 用同一线性映射（整图 bbox = xian.json bounds）

不重训、不换模型。AMC 的 inferred_rn 来自 DSFNet 全图预测裁到 region 子图。
"""
import os
import sys
import math
import pickle
from collections import defaultdict

import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, '..')))

from tptk.common.spatial_func import SPoint, distance  # noqa: E402
from tptk.common.road_network import UndirRoadNetwork  # noqa: E402
from tptk.common.mbr import MBR  # noqa: E402
from tptk.common.trajectory import Trajectory, STPoint, get_tid  # noqa: E402
import networkx as nx  # noqa: E402
from rtree import Rtree  # noqa: E402

import raster_to_shp as rts  # noqa: E402  (复用骨架化/构图)


# ============================================================
# 配置：xian.json + download_use_osm.py 切分参数
# ============================================================
XIAN_CFG = {
    'lat_min': 34.206385, 'lat_max': 34.279658,
    'lon_min': 108.917423, 'lon_max': 108.99286,
    'size': 400,  # region tile 像素边长
}
# DelvMap 整图（与 xian.json 同 bbox）
WHOLE_CFG = {
    'lat_min': 34.206385, 'lat_max': 34.279658,
    'lon_min': 108.917423, 'lon_max': 108.99286,
    'img_w': 5625, 'img_h': 6610,
}


def _grid_params(cfg=XIAN_CFG):
    """download_use_osm.py 的切分参数：dlat/dlon/lat_n/lon_n。"""
    size = cfg['size']
    dlat = size / 111111.0
    dlon = size / (111111.0 * math.cos(math.radians(cfg['lat_min'])))
    lat_n = math.ceil((cfg['lat_max'] - cfg['lat_min']) / dlat)
    lon_n = math.ceil((cfg['lon_max'] - cfg['lon_min']) / dlon)
    return dlat, dlon, lat_n, lon_n


def region_bbox(region_idx, cfg=XIAN_CFG):
    """region 编号 → (lat_st, lat_ed, lon_st, lon_ed)。NW-first (与 download_use_osm.py 一致)。

    编号规则 (download_use_osm.py NW-first): tile 从左上角(NW)开始行优先 TL→BR。
      i = idx // lon_n (行, i=0 = 最北行), j = idx % lon_n (列, j=0 = 最西列)。
    region i,j 的 lat 范围 [lat_max - (i+1)*dlat, lat_max - i*dlat] (i 增大向南),
                  lon 范围 [lon_min + j*dlon, + dlon] (j 增大向东)。
    注意 lat_st < lat_ed (st 是南边小纬度)。
    """
    dlat, dlon, lat_n, lon_n = _grid_params(cfg)
    i = region_idx // lon_n
    j = region_idx % lon_n
    lat_ed = cfg['lat_max'] - i * dlat          # 北边 (i=0 = lat_max)
    lat_st = cfg['lat_max'] - (i + 1) * dlat    # 南边
    lon_st = cfg['lon_min'] + j * dlon
    lon_ed = lon_st + dlon
    return lat_st, lat_ed, lon_st, lon_ed


# ============================================================
# 坐标映射 (线性, 与 didi_xian 一致)
# ============================================================
def rc_to_latlon(row, col, bbox, size=XIAN_CFG['size']):
    """region (row,col) → (lat,lon)。row=0 北(lat_ed), col=0 西(lon_st)。"""
    lat_st, lat_ed, lon_st, lon_ed = bbox
    lat = lat_ed - (row / size) * (lat_ed - lat_st)
    lon = lon_st + (col / size) * (lon_ed - lon_st)
    return lat, lon


def latlon_to_rc(lat, lon, bbox, size=XIAN_CFG['size']):
    """(lat,lon) → region (row,col)。逆变换。"""
    lat_st, lat_ed, lon_st, lon_ed = bbox
    row = (lat_ed - lat) / (lat_ed - lat_st) * size
    col = (lon - lon_st) / (lon_ed - lon_st) * size
    return row, col


def bbox_to_whole_pixel(bbox, whole_cfg=WHOLE_CFG):
    """region geo bbox → DelvMap 整图像素窗口 (px_st, py_st, px_ed, py_ed)。

    整图像素 ↔ WGS84 用同一线性映射: px = (lon-lon_min)/(lon_max-lon_min)*img_w,
    py = (lat_max-lat)/(lat_max-lat_min)*img_h (北在上, py=0 北)。
    """
    lat_st, lat_ed, lon_st, lon_ed = bbox
    px_st = (lon_st - whole_cfg['lon_min']) / (whole_cfg['lon_max'] - whole_cfg['lon_min']) * whole_cfg['img_w']
    px_ed = (lon_ed - whole_cfg['lon_min']) / (whole_cfg['lon_max'] - whole_cfg['lon_min']) * whole_cfg['img_w']
    py_st = (whole_cfg['lat_max'] - lat_ed) / (whole_cfg['lat_max'] - whole_cfg['lat_min']) * whole_cfg['img_h']
    py_ed = (whole_cfg['lat_max'] - lat_st) / (whole_cfg['lat_max'] - whole_cfg['lat_min']) * whole_cfg['img_h']
    return int(round(px_st)), int(round(py_st)), int(round(px_ed)), int(round(py_ed))


# ============================================================
# B. pickle 邻接字典 ↔ UndirRoadNetwork
# ============================================================
def pickle_to_rn(adj_dict, bbox):
    """didi_xian 邻接字典 {(row,col):[(row,col)...]} → UndirRoadNetwork (WGS84)。

    节点 (row,col) → (lat,lon) → node key (lng,lat) (与 AMC/tptk 约定一致)。
    每条边 coords=[SPoint(start), SPoint(end)], eid 递增, length=haversine 米。
    """
    g = nx.Graph()
    edge_spatial_idx = Rtree()
    edge_idx = {}
    eid = 0
    # 收集所有节点（含邻居）
    nodes = set(adj_dict.keys())
    for nbrs in adj_dict.values():
        nodes.update(nbrs)
    # 建节点
    for (row, col) in nodes:
        lat, lon = rc_to_latlon(row, col, bbox)
        g.add_node((lon, lat), pt=SPoint(lat, lon))
    # 建边（无向去重）
    seen = set()
    for u_rc, nbrs in adj_dict.items():
        for v_rc in nbrs:
            key = frozenset((u_rc, v_rc))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            u = (rc_to_latlon(u_rc[0], u_rc[1], bbox)[1], rc_to_latlon(u_rc[0], u_rc[1], bbox)[0])
            v = (rc_to_latlon(v_rc[0], v_rc[1], bbox)[1], rc_to_latlon(v_rc[0], v_rc[1], bbox)[0])
            sp_u = SPoint(u[1], u[0])
            sp_v = SPoint(v[1], v[0])
            coords = [sp_u, sp_v]
            data = {'eid': eid, 'coords': coords, 'length': distance(sp_u, sp_v)}
            g.add_edge(u, v, **data)
            mbr = MBR.cal_mbr(coords)
            edge_spatial_idx.insert(eid, (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
            edge_idx[eid] = (u, v)
            eid += 1
    return UndirRoadNetwork(g, edge_spatial_idx, edge_idx)


def rn_to_pickle(rn, bbox):
    """UndirRoadNetwork (WGS84) → didi_xian 邻接字典 {(row,col):[(row,col)...]}。

    AMC 输出的边 coords 是 SPoint(lat,lon) 列表。取每条边两端节点转 (row,col)。
    中间点丢失（AMC 边已简化，端点足够）。与 graph_gt.pickle 同构。
    """
    adj = defaultdict(list)
    seen = set()
    for u, v, data in rn.edges(data=True):
        # u/v 是 (lng,lat) tuple; 也可能 coords 有中间点, 用 coords 两端更准
        coords = data.get('coords')
        if coords and len(coords) >= 2:
            sp_u, sp_v = coords[0], coords[-1]
            lat_u, lon_u = sp_u.lat, sp_u.lng
            lat_v, lon_v = sp_v.lat, sp_v.lng
        else:
            lat_u, lon_u = u[1], u[0]
            lat_v, lon_v = v[1], v[0]
        row_u, col_u = latlon_to_rc(lat_u, lon_u, bbox)
        row_v, col_v = latlon_to_rc(lat_v, lon_v, bbox)
        # 量化到 0.001 像素避免浮点抖动产生重复节点
        nu = (round(row_u, 3), round(col_u, 3))
        nv = (round(row_v, 3), round(col_v, 3))
        if nu == nv:
            continue
        key = frozenset((nu, nv))
        if key in seen:
            continue
        seen.add(key)
        adj[nu].append(nv)
        adj[nv].append(nu)
    return dict(adj)


# ============================================================
# C. active_graph → trajs
# ============================================================
def active_graph_to_trajs(adj_dict, bbox):
    """active_graph 邻接字典 → list[Trajectory]。

    每条边 → 一段 Trajectory(pt_list=[STPoint(start), STPoint(end)])。
    AMC 的 adaptive_fuse 用 trajs 做 P1 轨迹投票 (get_max_trans_edge)。
    """
    from datetime import datetime, timedelta
    trajs = []
    seen = set()
    oid = 0
    base_time = datetime(2019, 1, 1)
    for u_rc, nbrs in adj_dict.items():
        for v_rc in nbrs:
            key = frozenset((u_rc, v_rc))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            lat_u, lon_u = rc_to_latlon(u_rc[0], u_rc[1], bbox)
            lat_v, lon_v = rc_to_latlon(v_rc[0], v_rc[1], bbox)
            t0 = base_time + timedelta(seconds=oid * 2)
            t1 = base_time + timedelta(seconds=oid * 2 + 1)
            pt_list = [STPoint(lat_u, lon_u, t0), STPoint(lat_v, lon_v, t1)]
            trajs.append(Trajectory(str(oid), get_tid(str(oid), pt_list), pt_list))
            oid += 1
    return trajs


# ============================================================
# D. DSFNet 整图预测 → region inferred_rn (线性坐标)
# ============================================================
def _pixel_edges_to_rn_linear(edges, bbox, size=XIAN_CFG['size'], simplify_eps_pixel=2.0):
    """骨架像素边 → UndirRoadNetwork，坐标用 region bbox 线性映射 (非墨卡托)。

    与 raster_to_shp.pixel_edges_to_rn 区别：pixel→geo 用 rc_to_latlon (线性)。
    像素坐标本身就是 region 内 (row,col)，直接转 (lat,lon)。
    """
    g = nx.Graph()
    edge_spatial_idx = Rtree()
    edge_idx = {}
    avail_eid = 0

    def _dp_pixel(pts, eps):
        if len(pts) <= 2:
            return pts
        keep = [False] * len(pts); keep[0] = keep[-1] = True
        stack = [(0, len(pts) - 1)]
        while stack:
            i, j = stack.pop()
            if j - i < 2:
                continue
            x1, y1 = pts[i]; x2, y2 = pts[j]
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
                stack.append((i, max_k)); stack.append((max_k, j))
        return [pts[k] for k in range(len(pts)) if keep[k]]

    for raw_edge in edges:
        pts_xy = [(x, y) for (y, x) in raw_edge]  # (row,col) -> (x=col,y=row) for DP
        pts_xy = _dp_pixel(pts_xy, simplify_eps_pixel)
        if len(pts_xy) < 2:
            continue
        coords_sp = []
        node_keys = []
        for (px, py) in pts_xy:  # px=col, py=row
            row, col = float(py), float(px)
            lat, lon = rc_to_latlon(row, col, bbox, size=size)
            coords_sp.append(SPoint(lat, lon))
            node_keys.append((lon, lat))
        u, v = node_keys[0], node_keys[-1]
        if u == v and len(node_keys) <= 2:
            continue
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
    return UndirRoadNetwork(g, edge_spatial_idx, edge_idx)


def dsfnet_region_inferred(region_idx, pred_dir, head='pred_traj_img',
                            whole_cfg=WHOLE_CFG, bin_thresh=128, min_obj=50,
                            simplify_eps=2.0, region_size=XIAN_CFG['size']):
    """DSFNet 全图预测 → region_idx 的 inferred_rn (WGS84, 线性坐标)。

    关键：DSFNet 整图原生 ~1.23 m/px，而 didi region 是 region_size×region_size
    像素覆盖同一 geo bbox (1.0 m/px @ size=400)。两者像素尺度不同。
    为让 inferred 与 partial/GT 共享同一套 region 栅格，先把裁出的整图像素窗
    resize 到 region_size×region_size 再骨架化 —— 此后骨架像素 (r,c)∈[0,region_size)
    直接就是 region 内 (row,col)，rc_to_latlon(size=region_size) 精确成立。
    """
    bbox = region_bbox(region_idx)
    px_st, py_st, px_ed, py_ed = bbox_to_whole_pixel(bbox, whole_cfg)
    # 1. 拼全图（缓存到模块级避免重跑）
    global _FULL_PRED_CACHE
    key = (pred_dir, head)
    if key not in _FULL_PRED_CACHE:
        _FULL_PRED_CACHE[key] = rts.stitch_full_pred(pred_dir, head=head)
    full = _FULL_PRED_CACHE[key]
    # 2. 裁 region 在整图里的像素窗口（跨整图边界时 clamp，左上对齐 region 原点）
    H, W = full.shape
    x0 = max(0, px_st); y0 = max(0, py_st)
    x1 = min(W, px_ed); y1 = min(H, py_ed)
    if x1 <= x0 or y1 <= y0:
        # region 完全在整图外（理论不会，bbox 一致），返回空图
        return UndirRoadNetwork(nx.Graph(), Rtree(), {})
    region_pred = full[y0:y1, x0:x1]
    # 3. resize 到 region_size×region_size，统一到 didi region 栅格 (1.0 m/px)
    if region_pred.shape[0] != region_size or region_pred.shape[1] != region_size:
        region_pred = cv2.resize(region_pred, (region_size, region_size),
                                 interpolation=cv2.INTER_LINEAR)
    # 4. 骨架化 + 4-邻接构图 (复用 raster_to_shp)
    binary, skeleton = rts.binarize_and_skeletonize(
        region_pred, bin_thresh=bin_thresh, close_ksize=3, min_obj=min_obj)
    if int(skeleton.sum()) == 0:
        return UndirRoadNetwork(nx.Graph(), Rtree(), {})
    edges = rts.skeleton_to_graph(skeleton)
    if not edges:
        return UndirRoadNetwork(nx.Graph(), Rtree(), {})
    # 5. 像素→WGS84 线性映射。resize 后 (r,c)∈[0,region_size) 即 region (row,col)。
    rn = _pixel_edges_to_rn_linear(edges, bbox, size=region_size, simplify_eps_pixel=simplify_eps)
    return rn


_FULL_PRED_CACHE = {}


# ============================================================
# 工具：加载 didi_xian pickle
# ============================================================
def load_didi_pickle(region_idx, kind, didi_root=None):
    """加载 didi_xian region 的 pickle。

    kind: 'partial' | 'gt' | 'active'
    """
    if didi_root is None:
        # DelvMap 本地 (datasets/didi/xian/ 已 copy 进来, 自含)
        didi_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'datasets', 'didi', 'xian', '2019_400')
    if kind == 'partial':
        path = os.path.join(didi_root, 'partial_component', f'region_{region_idx}_refine_gt_graph_partial.p')
    elif kind == 'gt':
        path = os.path.join(didi_root, f'region_{region_idx}_graph_gt.pickle')
    elif kind == 'active':
        path = os.path.join(didi_root, f'region_{region_idx}_active_graph.pickle')
    else:
        raise ValueError(kind)
    return pickle.load(open(path, 'rb'))


if __name__ == '__main__':
    # 自检：坐标往返 + resize 对齐验证
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=int, default=9)
    parser.add_argument('--pred_dir', default='results/delvmap_exp2/all_357/images_full')
    args = parser.parse_args()

    bbox = region_bbox(args.region)
    print(f'region_{args.region} bbox (lat_st,lat_ed,lon_st,lon_ed): {[round(x, 6) for x in bbox]}')
    # 坐标往返
    lat, lon = rc_to_latlon(0, 0, bbox)
    r, c = latlon_to_rc(lat, lon, bbox)
    print(f'rc(0,0)->latlon({lat:.6f},{lon:.6f})->rc({r:.4f},{c:.4f})  (往返误差应~0)')
    # 整图像素窗口 + 裁窗尺寸 vs region 栅格
    px_st, py_st, px_ed, py_ed = bbox_to_whole_pixel(bbox)
    print(f'整图像素窗口: x[{px_st},{px_ed}] y[{py_st},{py_ed}]  裁窗={px_ed-px_st}x{py_ed-py_st}'
          f'  (resize 到 {XIAN_CFG["size"]}x{XIAN_CFG["size"]} 后骨架化)')
    # pickle 加载
    p = load_didi_pickle(args.region, 'partial')
    g = load_didi_pickle(args.region, 'gt')
    print(f'partial nodes={len(p)} gt nodes={len(g)}')
    # pickle_to_rn 往返
    rn = pickle_to_rn(p, bbox)
    print(f'partial → rn: nodes={rn.number_of_nodes()} edges={rn.number_of_edges()}')
    adj_back = rn_to_pickle(rn, bbox)
    print(f'rn → pickle: nodes={len(adj_back)}  (原 partial nodes={len(p)})')
    # DSFNet inferred (验证 resize 对齐)
    inf = dsfnet_region_inferred(args.region, args.pred_dir)
    print(f'inferred → rn: edges={inf.number_of_edges()}')
