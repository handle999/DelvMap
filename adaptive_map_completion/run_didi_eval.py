"""didi_xian 端到端评估编排：partial → AMC(DSFNet inferred) → fused → samroad APLS/TOPO。

对每个 region:
  1. partial pickle → partial_rn (AMC existing_rn)
  2. active_graph pickle → trajs (AMC P1 轨迹)
  3. DSFNet 全图预测裁到 region → inferred_rn (AMC inferred_rn)
  4. AMC.adaptive_fuse(partial_rn, inferred_rn, trajs) → fused_rn
  5. fused_rn → 邻接字典 pickle → {work_dir}/graph/{region}.p
  6. 调 samroad metrics/eval.py --dataset didi_xian 评估 APLS+TOPO

用法：
    # 单 region 验证
    python adaptive_map_completion/run_didi_eval.py \
        --pred_dir results/delvmap_exp2/all_357/images_full \
        --regions 9 --work_dir didi_eval

    # 全 test split
    python adaptive_map_completion/run_didi_eval.py \
        --pred_dir results/delvmap_exp2/all_357/images_full \
        --work_dir didi_eval --split test

    # sanity: GT 自评 (验证评估链路, 应 APLS≈1.0)
    python adaptive_map_completion/run_didi_eval.py --regions 9 --work_dir didi_eval_sanity --pred_mode gt

    # sanity: partial 自评 (baseline 下限)
    python adaptive_map_completion/run_didi_eval.py --regions 9 --work_dir didi_eval_partial --pred_mode partial
"""
import os
import sys
import json
import argparse
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, '..')))

import didi_bridge as db  # noqa: E402
from adaptive_map_completion import DelvMapConnector  # noqa: E402
import pickle  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# 评估自含: metrics/ (eval.py+topo+apls) 与 datasets/didi/xian/ 已 copy 进 DelvMap,
# 不再依赖 samroad 项目。run_eval 调本地 metrics/eval.py。
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
PY = sys.executable
DIDI_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'didi', 'xian', '2019_400')
SPLIT_FILE = os.path.join(PROJECT_ROOT, 'datasets', 'didi', 'xian', 'data_split.json')

# samroad triage 配色 (BGR, 与 tools/viz_compare_samroad_p2cnet.py commit e1ed765 一致)
TRIAGE_EDGE = (15, 160, 253)    # RGB(253,160,15) 橙黄边
TRIAGE_NODE = (0, 255, 255)     # RGB(255,255,0) 黄节点
EDGE_THICK = 4
NODE_RADIUS = 4
SIZE = 400


def _fused_to_mask(fused_adj, size=SIZE):
    """fused 邻接字典 {(row,col):[(row,col)...]} → 二值 road mask (size×size, 0/255)。

    对齐 samroad mask/{idx}_road.png: road=255, 背景=0。
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    seen = set()
    for u_rc, nbrs in fused_adj.items():
        for v_rc in nbrs:
            key = frozenset((u_rc, v_rc))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            cv2.line(mask, (int(round(u_rc[1])), int(round(u_rc[0]))),
                     (int(round(v_rc[1])), int(round(v_rc[0]))), 255, 3, cv2.LINE_AA)
    return mask


def _save_viz(region_idx, fused_adj, work_dir, size=SIZE):
    """sat + fused graph 叠加 → {work_dir}/viz/{region}.png (triage 配色, 单图)。"""
    sat = cv2.imread(os.path.join(DIDI_ROOT, f'region_{region_idx}_sat.png'))
    if sat is None:
        return
    viz = sat.copy()
    seen = set()
    for u_rc, nbrs in fused_adj.items():
        for v_rc in nbrs:
            key = frozenset((u_rc, v_rc))
            if len(key) == 1 or key in seen:
                continue
            seen.add(key)
            cv2.line(viz, (int(round(u_rc[1])), int(round(u_rc[0]))),
                     (int(round(v_rc[1])), int(round(v_rc[0]))), TRIAGE_EDGE, EDGE_THICK, cv2.LINE_AA)
    for u_rc in fused_adj.keys():
        cv2.circle(viz, (int(round(u_rc[1])), int(round(u_rc[0]))), NODE_RADIUS, TRIAGE_NODE, -1, cv2.LINE_AA)
    viz_dir = os.path.join(work_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    cv2.imwrite(os.path.join(viz_dir, f'{region_idx}.png'), viz)


def run_one_region(region_idx, pred_dir, work_dir, pred_mode='amc',
                   head='pred_traj_img', min_trans_cnt=1, verbose=True):
    """跑单个 region 的完整链路，写 {work_dir}/graph/{region}.p。

    pred_mode:
      'amc'    : partial + DSFNet inferred → AMC fuse (主流程)
      'gt'     : 直接用 graph_gt.pickle 当 pred (sanity, 应≈1.0)
      'partial': 直接用 partial pickle 当 pred (baseline 下限)
    返回写入的 graph pickle 路径。
    """
    bbox = db.region_bbox(region_idx)
    graph_dir = os.path.join(work_dir, 'graph')
    os.makedirs(graph_dir, exist_ok=True)
    out_pkl = os.path.join(graph_dir, f'{region_idx}.p')

    if pred_mode == 'gt':
        adj = db.load_didi_pickle(region_idx, 'gt', DIDI_ROOT)
        pickle.dump(adj, open(out_pkl, 'wb'))
        if verbose:
            print(f'[region {region_idx}] GT mode: dumped gt pickle ({len(adj)} nodes)')
        return out_pkl
    if pred_mode == 'partial':
        adj = db.load_didi_pickle(region_idx, 'partial', DIDI_ROOT)
        pickle.dump(adj, open(out_pkl, 'wb'))
        if verbose:
            print(f'[region {region_idx}] partial mode: dumped partial pickle ({len(adj)} nodes)')
        return out_pkl

    # pred_mode == 'amc': 完整链路
    # 1. partial → rn (AMC existing_rn / base)
    partial_adj = db.load_didi_pickle(region_idx, 'partial', DIDI_ROOT)
    partial_rn = db.pickle_to_rn(partial_adj, bbox)
    if verbose:
        print(f'[region {region_idx}] partial_rn: {partial_rn.number_of_nodes()} nodes, '
              f'{partial_rn.number_of_edges()} edges')

    # partial 空 (采样删光) → 没有现有路网可补, 跳过该 region (评估时自动 SKIP,
    # 与 samroad/P2CNet 处理缺失 pred 一致)。
    if partial_rn.number_of_edges() == 0:
        print(f'[region {region_idx}] SKIP: partial empty (no base to complete)')
        return None

    # 2. trajs: didi_xian 没有 mm 轨迹文件, AMC 走纯 P2 空间就近合并 (trajs=[])。
    #    active_graph 是"轨迹走过的路"的图, 不是轨迹本身, 不应作为 trajs 输入。
    trajs = []
    if verbose:
        print(f'[region {region_idx}] trajs: 0 (no mm traj; AMC P2 spatial-only)')

    # 3. DSFNet → inferred_rn (AMC inferred_rn, 补全候选)
    inferred_rn = db.dsfnet_region_inferred(region_idx, pred_dir, head=head)
    if verbose:
        print(f'[region {region_idx}] inferred_rn (DSFNet): {inferred_rn.number_of_nodes()} nodes, '
              f'{inferred_rn.number_of_edges()} edges')
    if inferred_rn.number_of_edges() == 0:
        print(f'[region {region_idx}] WARN: DSFNet inferred empty; fused = partial only')
        fused_rn = partial_rn
    else:
        # 4. AMC fuse: 把 inferred_rn 里 partial 没有的路 (32m 投影检查) 增量拼到 partial,
        #    交叉路口连接点用 trajs 投票仲裁 (无 trajs 走 P2 空间就近)。
        # out_compressed=False: 评估时保留原始节点密度 (compress 会合并 degree=2 中间节点,
        # 使边长翻倍、节点稀疏, 导致 APLS 控制点与 partial/GT 对不齐, 评估失真)。
        mc = DelvMapConnector(out_compressed=False, min_trans_cnt=min_trans_cnt)
        fused_rn = mc.adaptive_fuse(partial_rn, inferred_rn, trajs)
    if verbose:
        print(f'[region {region_idx}] fused_rn: {fused_rn.number_of_nodes()} nodes, '
              f'{fused_rn.number_of_edges()} edges')

    # 5. fused → pickle
    fused_adj = db.rn_to_pickle(fused_rn, bbox)
    pickle.dump(fused_adj, open(out_pkl, 'wb'))
    if verbose:
        print(f'[region {region_idx}] → {out_pkl} ({len(fused_adj)} nodes)')

    # 6. 保存产物 (仅 amc 模式, 对齐 samroad infer 目录结构):
    #    mask/{region}_road.png — fused 渲染的二值 road mask (partial 是 input 不保存)
    #    viz/{region}.png       — sat + fused 叠加 (triage 配色)
    mask_dir = os.path.join(work_dir, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    cv2.imwrite(os.path.join(mask_dir, f'{region_idx}_road.png'), _fused_to_mask(fused_adj))
    _save_viz(region_idx, fused_adj, work_dir)
    return out_pkl


def run_eval(work_dir, metric='all', workers=None):
    """调本地 metrics/eval.py 评估 {work_dir}/graph/*.p。

    metrics/ 与 datasets/didi/xian/ 已 copy 进 DelvMap, 无跨项目依赖、无软链接。
    eval.py 的 --dir 需相对 DelvMap 项目根 (其内部用 ../{dir}/graph/{idx}.p, cwd=metrics/)。
    """
    abs_work = os.path.abspath(work_dir)
    rel_dir = os.path.relpath(abs_work, PROJECT_ROOT)  # 相对 DelvMap 根
    eval_py = os.path.join(PROJECT_ROOT, 'metrics', 'eval.py')
    cmd = [PY, eval_py, '--dataset', 'didi_xian', '--dir', rel_dir, '--metric', metric]
    if workers is not None:
        cmd += ['--workers', str(workers)]
    print(f'\n[eval] {" ".join(cmd)}')
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print('[eval STDERR]', res.stderr[-2000:])
    return res.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', default='results/delvmap_exp2/all_357/images_full',
                   help='DSFNet infer_all 输出目录')
    p.add_argument('--regions', default=None, help='逗号分隔的 region 编号, 如 9,17,92')
    p.add_argument('--split', default=None, choices=['test', 'train', 'val'],
                   help='跑 data_split.json 的某个 split (与 --regions 互斥)')
    p.add_argument('--work_dir', default='didi_eval', help='工作/输出目录')
    p.add_argument('--pred_mode', default='amc', choices=['amc', 'gt', 'partial'],
                   help='amc=完整链路; gt=GT自评; partial=partial自评')
    p.add_argument('--head', default='pred_traj_img',
                   choices=['pred_traj_img', 'pred_src_traj_img'])
    p.add_argument('--min_trans_cnt', type=int, default=1)
    p.add_argument('--no_eval', action='store_true', help='只生成 graph pickle, 不跑 samroad 评估')
    p.add_argument('--metric', default='all', choices=['all', 'apls', 'topo'])
    p.add_argument('--workers', type=int, default=None)
    args = p.parse_args()

    # 解析 region 列表
    if args.regions:
        regions = [int(x) for x in args.regions.split(',')]
    elif args.split:
        split = json.load(open(SPLIT_FILE))
        regions = [int(x) for x in split.get(args.split, [])]
    else:
        print('必须指定 --regions 或 --split')
        sys.exit(1)

    print(f'=== regions: {len(regions)} | pred_mode: {args.pred_mode} | work_dir: {args.work_dir} ===')

    # 清理旧的 graph pickle
    graph_dir = os.path.join(args.work_dir, 'graph')
    if os.path.exists(graph_dir):
        for f in os.listdir(graph_dir):
            if f.endswith('.p'):
                os.remove(os.path.join(graph_dir, f))

    for ridx in regions:
        print(f'\n----- region {ridx} -----')
        try:
            run_one_region(ridx, args.pred_dir, args.work_dir,
                           pred_mode=args.pred_mode, head=args.head,
                           min_trans_cnt=args.min_trans_cnt)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'[region {ridx}] FAILED: {e}')

    if not args.no_eval:
        run_eval(args.work_dir, metric=args.metric, workers=args.workers)


if __name__ == '__main__':
    main()
