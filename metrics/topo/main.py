import numpy as np
import math
import sys
import pickle
import graph as splfy
import topo as topo
import json
import os

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-graph_gt', default='', action='store', dest='graph_gt', type=str,
                    help='ground truth graph (in xy coordinate)')

parser.add_argument('-graph_prop', default='', action='store', dest='graph_prop', type=str,
                    help='proposed graph (in xy coordinate)')

parser.add_argument('-output', default='', action='store', dest='output', type=str,
                    help="outputfile with '.txt' as suffix")

parser.add_argument('-matching_threshold', action='store', dest='matching_threshold', type=float,
                    help='topo marble-hole matching distance', required=False, default=0.00010)

parser.add_argument('-interval', action='store', dest='topo_interval', type=float,
                    help='topo marble-hole interval', required=False, default=0.00005)

parser.add_argument('-savedir', type=str, required=True)

parser.add_argument('-dataset', type=str, default='spacenet',
                    choices=['cityscale', 'spacenet', 'didi_xian', 'didi'],
                    help="dataset type to determine path patterns; 'didi' is legacy alias of 'didi_xian'")

parser.add_argument('-topo_radius', type=float, default=None,
                    help='TOPO propagation radius (default: 0.0015 for spacenet, 0.003 for cityscale)')

args = parser.parse_args()
if args.dataset == 'didi':
    print("[WARN] dataset='didi' is deprecated; use 'didi_xian' instead.")
    args.dataset = 'didi_xian'
print(args)

# Dataset-specific configuration
if args.dataset == 'cityscale':
    test_indices = [8, 9, 19, 28, 29, 39, 48, 49, 59, 68, 69, 79, 88, 89, 99,
                    108, 109, 119, 128, 129, 139, 148, 149, 159, 168, 169, 179]
    gt_pattern = '../datasets/cityscale/20cities/region_{}_graph_gt.pickle'
    pred_pattern = '../{savedir}/graph/{idx}.p'
    topo_r = args.topo_radius if args.topo_radius else 0.003
elif args.dataset == 'spacenet':
    with open('../datasets/spacenet/data_split.json', 'r') as jf:
        test_indices = json.load(jf)['test']
    gt_pattern = '../datasets/spacenet/RGB_1.0_meter/{}__gt_graph.p'
    pred_pattern = '../{savedir}/graph/{idx}.p'
    topo_r = args.topo_radius if args.topo_radius else 0.0015
elif args.dataset == 'didi_xian':
    with open('../datasets/didi/xian/data_split.json', 'r') as jf:
        test_indices = json.load(jf)['test']
    gt_pattern = '../datasets/didi/xian/2019_400/region_{}_graph_gt.pickle'
    pred_pattern = '../{savedir}/graph/{idx}.p'
    topo_r = args.topo_radius if args.topo_radius else 0.0015

lat_top_left = 41.0
lon_top_left = -71.0
min_lat = 41.0
max_lon = -71.0

for tile_idx in test_indices:

    graph_prop_path = pred_pattern.format(savedir=args.savedir, idx=tile_idx)
    graph_gt_path = gt_pattern.format(tile_idx)
    output_path = '../{savedir}/results/topo/{idx}.txt'.format(savedir=args.savedir, idx=tile_idx)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(graph_prop_path):
        continue

    map1 = pickle.load(open(graph_gt_path, "rb"))
    map2 = pickle.load(open(graph_prop_path, "rb"))


    def xy2latlon(x, y):
        lat = lat_top_left - x * 1.0 / 111111.0
        lon = lon_top_left + (y * 1.0 / 111111.0) / math.cos(math.radians(lat_top_left))

        return lat, lon


    def create_graph(m):
        global min_lat
        global max_lon

        graph = splfy.RoadGraph()

        nid = 0
        idmap = {}

        def getid(k, idmap):

            if k in idmap:
                return idmap[k]

            idmap[k] = nid
            nid += 1

            return idmap[k]

        for k, v in m.items():
            n1 = k

            lat1, lon1 = xy2latlon(n1[0], n1[1])

            if lat1 < min_lat:
                min_lat = lat1

            if lon1 > max_lon:
                max_lon = lon1

            for n2 in v:
                lat2, lon2 = xy2latlon(n2[0], n2[1])

                if n1 in idmap:
                    id1 = idmap[n1]
                else:
                    id1 = nid
                    idmap[n1] = nid
                    nid = nid + 1

                if n2 in idmap:
                    id2 = idmap[n2]
                else:
                    id2 = nid
                    idmap[n2] = nid
                    nid = nid + 1

                graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)

        graph.ReverseDirectionLink()

        for node in graph.nodes.keys():
            graph.nodeScore[node] = 100

        for edge in graph.edges.keys():
            graph.edgeScore[edge] = 100

        return graph

    graph_gt = create_graph(map1)
    graph_prop = create_graph(map2)

    print("load gt/prop graphs")

    region = [min_lat - 300 * 1.0 / 111111.0, lon_top_left - 500 * 1.0 / 111111.0,
              lat_top_left + 300 * 1.0 / 111111.0, max_lon + 500 * 1.0 / 111111.0]

    graph_gt.region = region
    graph_prop.region = region

    losm = topo.TOPOGenerateStartingPoints(graph_gt, region=region, image="NULL", check=False, direction=False, metaData=None)

    lmap = topo.TOPOGeneratePairs(graph_prop, graph_gt, losm, threshold=0.00010, region=region)

    topoResult = topo.TOPOWithPairs(graph_prop, graph_gt, lmap, losm, r=topo_r,
                                     step=args.topo_interval, threshold=args.matching_threshold,
                                     outputfile=output_path, one2oneMatching=True, metaData=None)

    print('=========', output_path, '==================')

    pickle.dump([losm, topoResult, region], open(output_path.replace('txt', 'topo.p'), 'wb'))
