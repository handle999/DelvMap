import os
import sys
import json
import math
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# 导入同目录下的核心算法库
import graph as splfy
import topo as topo

# 修复雷点3：将写死的常量提取，隔离环境
LAT_TOP_LEFT = 41.0
LON_TOP_LEFT = -71.0

def xy2latlon(x, y):
    lat = LAT_TOP_LEFT - x * 1.0 / 111111.0
    lon = LON_TOP_LEFT + (y * 1.0 / 111111.0) / math.cos(math.radians(LAT_TOP_LEFT))
    return lat, lon

def create_graph(m):
    """
    修复雷点1：去掉 global，将极值变量变为局部变量，防止多张图互相污染
    """
    graph = splfy.RoadGraph()
    nid = 0
    idmap = {}
    
    min_lat_local = LAT_TOP_LEFT
    max_lon_local = LON_TOP_LEFT

    for k, v in m.items():
        n1 = k
        lat1, lon1 = xy2latlon(n1[0], n1[1])

        # 更新本图自己的边界极值
        if lat1 < min_lat_local: min_lat_local = lat1
        if lon1 > max_lon_local: max_lon_local = lon1

        for n2 in v:
            lat2, lon2 = xy2latlon(n2[0], n2[1])

            if n1 in idmap:
                id1 = idmap[n1]
            else:
                id1 = nid
                idmap[n1] = nid
                nid += 1

            if n2 in idmap:
                id2 = idmap[n2]
            else:
                id2 = nid
                idmap[n2] = nid
                nid += 1

            graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)
    
    graph.ReverseDirectionLink()

    for node in graph.nodes.keys():
        graph.nodeScore[node] = 100
    for edge in graph.edges.keys():
        graph.edgeScore[edge] = 100

    return graph, min_lat_local, max_lon_local

def process_single_tile(tile_idx, savedir, interval, matching_threshold, dataset='spacenet'):
    """独立的单进程工作单元"""
    graph_prop_path = f'../{savedir}/graph/{tile_idx}.p'
    # 按 dataset 选 GT 路径模板, 与 metrics/topo/main.py 保持一致
    if dataset == 'cityscale':
        graph_gt_path = f'../datasets/cityscale/20cities/region_{tile_idx}_graph_gt.pickle'
    elif dataset == 'spacenet':
        graph_gt_path = f'../datasets/spacenet/RGB_1.0_meter/{tile_idx}__gt_graph.p'
    elif dataset == 'didi_xian':
        graph_gt_path = f'../datasets/didi/xian/2019_400/region_{tile_idx}_graph_gt.pickle'
    else:
        return f"ERROR: unknown dataset '{dataset}'"
    output_txt = f'../{savedir}/results/topo/{tile_idx}.txt'
    
    if not os.path.exists(graph_prop_path):
        return f"SKIP: Missing {graph_prop_path}"
        
    try:
        with open(graph_gt_path, "rb") as f:
            map1 = pickle.load(f)
        with open(graph_prop_path, "rb") as f:
            map2 = pickle.load(f)

        graph_gt, gt_min_lat, gt_max_lon = create_graph(map1)
        graph_prop, prop_min_lat, prop_max_lon = create_graph(map2)

        # 获取合并后的边界框
        final_min_lat = min(gt_min_lat, prop_min_lat)
        final_max_lon = max(gt_max_lon, prop_max_lon)

        region = [
            final_min_lat - 300 * 1.0/111111.0, 
            LON_TOP_LEFT - 500 * 1.0/111111.0, 
            LAT_TOP_LEFT + 300 * 1.0/111111.0, 
            final_max_lon + 500 * 1.0/111111.0
        ]

        graph_gt.region = region
        graph_prop.region = region

        losm = topo.TOPOGenerateStartingPoints(graph_gt, region=region, image="NULL", check=False, direction=False, metaData=None)
        lmap = topo.TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold=0.00010, region=region)

        r = 0.00150 
        topoResult = topo.TOPOWithPairs(
            graph_prop, graph_gt, lmap, losm, 
            r=r, step=interval, threshold=matching_threshold, 
            outputfile=output_txt, one2oneMatching=True, metaData=None
        )

        # 修复雷点3：砍掉了无用的巨型 .topo.p 文件的写入，榨干 I/O 性能
        return "SUCCESS"
    except Exception as e:
        return f"Exception ({tile_idx}): {str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-savedir', type=str, required=True)
    parser.add_argument('-workers', type=int, default=1)
    parser.add_argument('-matching_threshold', type=float, default=0.00010)
    parser.add_argument('-interval', type=float, default=0.00005)
    parser.add_argument('-dataset', type=str, default='spacenet',
                        choices=['cityscale', 'spacenet', 'didi_xian', 'didi'],
                        help="Dataset type; 'didi' is legacy alias of 'didi_xian'")
    args = parser.parse_args()
    if args.dataset == 'didi':
        print("[WARN] dataset='didi' is deprecated; use 'didi_xian' instead.")
        args.dataset = 'didi_xian'

    out_dir = f'../{args.savedir}/results/topo'
    os.makedirs(out_dir, exist_ok=True)

    # 并发前清理旧文件
    for f in os.listdir(out_dir):
        if f.endswith('.txt'):
            os.remove(os.path.join(out_dir, f))

    # 按 dataset 选测试集列表 (与 main.py 保持一致)
    if args.dataset == 'cityscale':
        # 注: cityscale 的测试 ID 是固定的整数列表 (与训练脚本里的逻辑一致)
        tile_list = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99,
                     108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
    elif args.dataset == 'spacenet':
        with open('../datasets/spacenet/data_split.json', 'r') as jf:
            tile_list = json.load(jf)['test']
    elif args.dataset == 'didi_xian':
        with open('../datasets/didi/xian/data_split.json', 'r') as jf:
            tile_list = json.load(jf)['test']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    # 重点：纯 Python CPU 密集型任务，必须用 ProcessPoolExecutor 多进程
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_tile, tile, args.savedir,
                            args.interval, args.matching_threshold,
                            args.dataset): tile
            for tile in tile_list
        }
        
        if has_tqdm:
            iterable = tqdm(as_completed(futures), total=len(tile_list), desc="TOPO Progress")
        else:
            print("TOPO Progress: ", end="", flush=True)
            iterable = as_completed(futures)
            
        errors = []
        for future in iterable:
            res = future.result()
            if res != "SUCCESS":
                errors.append(res)
            elif not has_tqdm:
                print(".", end="", flush=True)
                
    if not has_tqdm:
        print()
        
    if errors:
        print(f"\n[DEBUG] Encountered {len(errors)} errors. Showing top 5:")
        for err in list(set(errors))[:5]:
            print(f" - {err}")
            