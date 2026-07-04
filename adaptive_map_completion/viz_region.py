"""didi_xian region 可视化调试工具：partial vs fused 单图保存。

配色对齐 samroad 官方对比图 (tools/viz_compare_samroad_p2cnet.py, commit e1ed765):
  - graph edge: triage 橙黄 RGB(253,160,15) = BGR(15,160,253), thickness=4
  - graph node: 黄 RGB(255,255,0) = BGR(0,255,255), radius=4
  - cv2.LINE_AA

单图保存 (不拼接), 对齐 samroad infer 的 viz/{idx}.png 命名:
  {out_dir}/{idx}_partial.png  — sat + partial (triage 配色)
  {out_dir}/{idx}_fused.png    — sat + fused   (triage 配色)
  {out_dir}/{idx}_diff.png     — sat + partial(黄) vs fused(绿) 叠加

注: 批量评估时的 mask/viz 由 run_didi_eval.py 统一生成 (mask/{idx}_road.png, viz/{idx}.png),
    本脚本仅用于单 region 调试。

用法：
    python adaptive_map_completion/viz_region.py --region 92 \
        --pred_dir results/delvmap_exp2/all_357/images_full \
        --out_dir experiments/viz
"""
import os
import sys
import argparse

import numpy as np
import cv2

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, '..')))

import didi_bridge as db  # noqa: E402

DIDI_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'datasets', 'didi', 'xian', '2019_400')
SIZE = 400

# samroad triage 配色 (BGR)
TRIAGE_EDGE = (15, 160, 253)    # 橙黄边
TRIAGE_NODE = (0, 255, 255)     # 黄节点
EDGE_THICK = 4
NODE_RADIUS = 4
DIFF_PARTIAL = (0, 255, 255)    # 黄 = partial
DIFF_FUSED = (0, 255, 0)        # 绿 = fused (粗, 盖重合)


def _adj_to_edges(adj):
    seen = set()
    edges = []
    for u_rc, nbrs in adj.items():
        for v_rc in nbrs:
            key = frozenset((u_rc, v_rc))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            edges.append((u_rc, v_rc))
    return edges


def _rn_to_edges(rn, bbox):
    edges = []
    for u, v, data in rn.edges(data=True):
        coords = data.get('coords')
        if coords and len(coords) >= 2:
            sp_u, sp_v = coords[0], coords[-1]
        else:
            sp_u, sp_v = u, v
        r1, c1 = db.latlon_to_rc(sp_u.lat, sp_u.lng, bbox)
        r2, c2 = db.latlon_to_rc(sp_v.lat, sp_v.lng, bbox)
        edges.append(((r1, c1), (r2, c2)))
    return edges


def _draw_graph(img, edges, edge_color=TRIAGE_EDGE, node_color=TRIAGE_NODE):
    """samroad triage 风格: 边 + 节点, LINE_AA。"""
    for (r1, c1), (r2, c2) in edges:
        cv2.line(img, (int(round(c1)), int(round(r1))),
                 (int(round(c2)), int(round(r2))), edge_color, EDGE_THICK, cv2.LINE_AA)
    nodes = set()
    for (r1, c1), (r2, c2) in edges:
        nodes.add((int(round(r1)), int(round(c1))))
        nodes.add((int(round(r2)), int(round(c2))))
    for (r, c) in nodes:
        cv2.circle(img, (c, r), NODE_RADIUS, node_color, -1, cv2.LINE_AA)
    return img


def viz_region(region_idx, pred_dir, out_dir, head='pred_traj_img', run_amc=True,
               min_trans_cnt=1):
    """生成 region 的 partial/fused 单图 + diff。"""
    os.makedirs(out_dir, exist_ok=True)
    bbox = db.region_bbox(region_idx)
    sat = cv2.imread(os.path.join(DIDI_ROOT, f'region_{region_idx}_sat.png'))
    if sat is None:
        print(f'[viz] region_{region_idx}: sat not found, skip')
        return

    part_adj = db.load_didi_pickle(region_idx, 'partial', DIDI_ROOT)
    part_edges = _adj_to_edges(part_adj) if part_adj else []

    fused_edges = []
    if run_amc and part_adj:
        partial_rn = db.pickle_to_rn(part_adj, bbox)
        if partial_rn.number_of_edges() > 0:
            inferred_rn = db.dsfnet_region_inferred(region_idx, pred_dir, head=head)
            if inferred_rn.number_of_edges() > 0:
                from adaptive_map_completion import DelvMapConnector
                mc = DelvMapConnector(out_compressed=False, min_trans_cnt=min_trans_cnt)
                fused_rn = mc.adaptive_fuse(partial_rn, inferred_rn, [])
                fused_edges = _rn_to_edges(fused_rn, bbox)
            else:
                fused_edges = part_edges

    # 单图: partial (sat + partial, triage 配色)
    p_partial = sat.copy()
    _draw_graph(p_partial, part_edges)
    cv2.imwrite(os.path.join(out_dir, f'{region_idx}_partial.png'), p_partial)
    print(f'[viz] {out_dir}/{region_idx}_partial.png')

    # 单图: fused (sat + fused, triage 配色)
    p_fused = sat.copy()
    _draw_graph(p_fused, fused_edges)
    cv2.imwrite(os.path.join(out_dir, f'{region_idx}_fused.png'), p_fused)
    print(f'[viz] {out_dir}/{region_idx}_fused.png')

    # diff: partial(黄) vs fused(绿) 叠加
    diff = sat.copy()
    for (r1, c1), (r2, c2) in part_edges:
        cv2.line(diff, (int(round(c1)), int(round(r1))),
                 (int(round(c2)), int(round(r2))), DIFF_PARTIAL, 2, cv2.LINE_AA)
    for (r1, c1), (r2, c2) in fused_edges:
        cv2.line(diff, (int(round(c1)), int(round(r1))),
                 (int(round(c2)), int(round(r2))), DIFF_FUSED, 3, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f'{region_idx}_diff.png'), diff)
    print(f'[viz] {out_dir}/{region_idx}_diff.png')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--region', type=int, required=True)
    p.add_argument('--pred_dir', default='results/delvmap_exp2/all_357/images_full')
    p.add_argument('--head', default='pred_traj_img')
    p.add_argument('--out_dir', default='experiments/viz')
    p.add_argument('--no_amc', action='store_true')
    args = p.parse_args()
    viz_region(args.region, args.pred_dir, args.out_dir, head=args.head,
               run_amc=not args.no_amc)


if __name__ == '__main__':
    main()
