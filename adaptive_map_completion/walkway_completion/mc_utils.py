from tptk.common.trajectory import Trajectory, get_tid
from tptk.common.spatial_func import distance
from tptk.segmentation import Segmentation
from tptk.common.road_network import UndirRoadNetwork, load_rn_shp
from tptk.common.mbr import MBR
import networkx as nx
from rtree import Rtree
import copy


def cal_hausdorff_distance(pt_list_1, pt_list_2):
    forward_min_dist_list = []
    for pt_1 in pt_list_1:
        min_dist = min([distance(pt_1, pt_2) for pt_2 in pt_list_2])
        forward_min_dist_list.append(min_dist)
    forward_max_dist = max(forward_min_dist_list)
    backward_min_dist_list = []
    for pt_2 in pt_list_2:
        min_dist = min([distance(pt_2, pt_1) for pt_1 in pt_list_1])
        backward_min_dist_list.append(min_dist)
    backward_max_dist = max(backward_min_dist_list)
    h_dist = max(forward_max_dist, backward_max_dist)
    return h_dist


def line_2_wkt(pt_list):
    wkt = 'LINESTRING ('
    for pt in pt_list:
        wkt += '{} {}, '.format(pt.lng, pt.lat)
    wkt = wkt[:-2] + ')'
    return wkt


def get_unmatched_trajs(mm_traj, match_tolerance, eid_mapping=None):
    # list of tuples: unmatched_traj,pre_matched_eid,nxt_matched_eid
    # 具体格式：[([SPoint, ..], int, int), ...]
    unmatched_trajs = []
    tmp_unmatched_pt_list = []
    pre_pt_matched = False
    pre_matched_eid = None
    for mm_pt in mm_traj.pt_list:
        # 当前点匹配失败
        if mm_pt.data['candi_pt'] is None or mm_pt.data['candi_pt'].error > match_tolerance:
            tmp_unmatched_pt_list.append(mm_pt)
            pre_pt_matched = False
        # 当前点匹配成功
        else:
            # 前一个点匹配失败
            if not pre_pt_matched:
                # 构建一段未匹配轨迹
                if len(tmp_unmatched_pt_list) > 0:
                    unmatched_trajs.append((Trajectory(mm_traj.oid, get_tid(mm_traj.oid, tmp_unmatched_pt_list),
                                                      tmp_unmatched_pt_list),
                                            pre_matched_eid,
                                            mm_pt.data['candi_pt'].eid if eid_mapping is None else eid_mapping[mm_pt.data['candi_pt'].eid]))
                # clear the cache
                tmp_unmatched_pt_list = []
            # 更改前驱匹配状态
            pre_pt_matched = True
            pre_matched_eid = mm_pt.data['candi_pt'].eid if eid_mapping is None else eid_mapping[mm_pt.data['candi_pt'].eid]
    # 处理最后一段
    if len(tmp_unmatched_pt_list) > 0:
        unmatched_trajs.append((Trajectory(mm_traj.oid, get_tid(mm_traj.oid, tmp_unmatched_pt_list),
                                           tmp_unmatched_pt_list), pre_matched_eid, None))
    return unmatched_trajs


class DistanceSegmentation(Segmentation):
    def __init__(self, max_dist):
        super(Segmentation, self).__init__()
        self.max_dist = max_dist

    def segment(self, traj):
        segmented_traj_list = []
        pt_list = traj.pt_list
        if len(pt_list) <= 1:
            return []
        oid = traj.oid
        pre_pt = pt_list[0]
        partial_pt_list = [pre_pt]
        for cur_pt in pt_list[1:]:
            dist = distance(cur_pt, pre_pt)
            if dist <= self.max_dist:
                partial_pt_list.append(cur_pt)
            else:
                if len(partial_pt_list) > 1:
                    segmented_traj = Trajectory(oid, get_tid(oid, partial_pt_list), partial_pt_list)
                    segmented_traj_list.append(segmented_traj)
                partial_pt_list = [cur_pt]
            pre_pt = cur_pt
        if len(partial_pt_list) > 1:
            segmented_traj = Trajectory(oid, get_tid(oid, partial_pt_list), partial_pt_list)
            segmented_traj_list.append(segmented_traj)
        return segmented_traj_list


def make_directed_rn_undirected(directed_rn):
    """
    返回有向图id和无向图id的mapping关系，无向图
    Note:对于无向图来说，无法保证u为coords[0],v为coords[1],需要check后续代码，特别是利用到coords的地方
    """
    g = nx.Graph()
    edge_spatial_idx = Rtree()
    edge_idx = {}
    dg_eid2g_eid = {}

    # add nodes
    for n, data in directed_rn.nodes(data=True):
        new_data = copy.deepcopy(data)
        g.add_node(n, **new_data)
    # add edges
    for u, v, data in directed_rn.edges(data=True):
        if g.has_edge(u, v):
            dg_eid2g_eid[data['eid']] = g.get_edge_data(u, v)['eid']
        else:
            mbr = MBR.cal_mbr(data['coords'])
            new_data = copy.deepcopy(data)
            g.add_edge(u, v, **new_data)
            edge_spatial_idx.insert(new_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
            edge_idx[new_data['eid']] = (u, v)
            dg_eid2g_eid[data['eid']] = data['eid']
    print('# of nodes:{}'.format(g.number_of_nodes()))
    print('# of edges:{}'.format(g.number_of_edges()))
    return UndirRoadNetwork(g, edge_spatial_idx, edge_idx), dg_eid2g_eid


if __name__ == '__main__':
    # 调用undirected后在对路网做索引会导致对向eid数据的丢时，所以在
    sta_ids = ['571']
    base_dir = '/Users/sjruan/Downloads/walkway_completion_7_mm_res_all/'
    sta_rn_path_template = base_dir + 'rn-osm-wgs/rn-{}-220920'
    for sta_id in sta_ids:
        sta_rn = load_rn_shp(sta_rn_path_template.format(sta_id))
        sta_rn_undirected, dg_eid2g_eid = make_directed_rn_undirected(sta_rn)
        print()
