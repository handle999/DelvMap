"""AdaMap 顶层调度：把 OSM 现有路网 + DSFNet 推断路网 (+ 可选 mm 轨迹) 融合。

最小化用法（无轨迹，纯空间合并 P2 路径，先把管线打通）:
    python adaptive_map_completion/run_adamap.py \
        --existing_rn dataset/osm/rn-comp-xa-190101-didi \
        --inferred_rn inferred_rn_xa \
        --out_dir fused_rn_xa

完整用法（带 map-matched 轨迹，启用 P1 轨迹支撑接边）:
    python adaptive_map_completion/run_adamap.py \
        --existing_rn dataset/osm/rn-comp-xa-190101-didi \
        --inferred_rn inferred_rn_xa \
        --mm_traj_dir <DIR_OF_PER_COURIER_MM_TXT> \
        --out_dir fused_rn_xa
"""
import os
import sys
import argparse

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from tptk.common.road_network import load_rn_shp, store_rn_shp  # noqa: E402

from adaptive_map_completion import DelvMapConnector, obtain_segmented_trajs  # noqa: E402
from walkway_completion.mc_utils import make_directed_rn_undirected  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--existing_rn', required=True, help='OSM shapefile dir (含 edges.shp)')
    p.add_argument('--inferred_rn', required=True, help='DSFNet 推断 shapefile dir')
    p.add_argument('--mm_traj_dir', default=None,
                   help='已 map-match 的轨迹根目录 (per-courier 子目录)；为空则跳过 P1，纯走 P2 空间合并')
    p.add_argument('--out_dir', required=True, help='融合后 shapefile 输出目录')
    p.add_argument('--min_trans_cnt', type=int, default=1,
                   help='P1 轨迹支撑阈值；--mm_traj_dir 提供时生效')
    p.add_argument('--traj_seg_dist', type=int, default=20)
    p.add_argument('--no_compress', action='store_true', help='不做 compress_rn 后处理')
    args = p.parse_args()

    print(f"[1/4] Loading existing rn from {args.existing_rn} ...")
    existing_rn = load_rn_shp(args.existing_rn, is_directed=True)
    existing_rn, _ = make_directed_rn_undirected(existing_rn)

    print(f"\n[2/4] Loading inferred rn from {args.inferred_rn} ...")
    inferred_rn = load_rn_shp(args.inferred_rn, is_directed=True)
    inferred_rn, _ = make_directed_rn_undirected(inferred_rn)

    if args.mm_traj_dir:
        print(f"\n[3/4] Loading mm trajectories from {args.mm_traj_dir} ...")
        trajs = obtain_segmented_trajs(args.mm_traj_dir, traj_seg_dist=args.traj_seg_dist)
        print(f"  → {len(trajs)} segmented trajectories")
    else:
        print(f"\n[3/4] No --mm_traj_dir provided; running spatial-only fusion (P2).")
        trajs = []

    print(f"\n[4/4] Adaptive fusion ...")
    mc = DelvMapConnector(out_compressed=(not args.no_compress),
                          min_trans_cnt=args.min_trans_cnt)
    fused_rn = mc.adaptive_fuse(existing_rn, inferred_rn, trajs)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\nWriting fused rn to {args.out_dir} ...")
    store_rn_shp(fused_rn, args.out_dir)
    print(f"\n[done] fused → {args.out_dir}")


if __name__ == '__main__':
    main()
