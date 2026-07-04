import os
import networkx as nx
from rtree import Rtree
from osgeo import ogr, osr
from .spatial_func import SPoint, distance
from .mbr import MBR
import copy


class UndirRoadNetwork(nx.Graph):
    def __init__(self, g, edge_spatial_idx, edge_idx):
        super(UndirRoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.edge_idx = edge_idx

    def to_directed(self, as_view=False):
        """
        new edge will have new eid, and each original edge will have two edge with reversed coords
        :return:
        """
        assert as_view is False, "as_view is not supported"
        avail_eid = max([eid for u, v, eid in self.edges.data(data='eid')]) + 1
        g = nx.DiGraph()
        edge_spatial_idx = Rtree()
        edge_idx = {}
        # add nodes
        for n, data in self.nodes(data=True):
            new_data = copy.deepcopy(data)
            g.add_node(n, **new_data)
        # add edges
        for u, v, data in self.edges(data=True):
            mbr = MBR.cal_mbr(data['coords'])
            # add forward edge
            forward_data = copy.deepcopy(data)
            g.add_edge(u, v, **forward_data)
            edge_spatial_idx.insert(forward_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
            edge_idx[forward_data['eid']] = (u, v)
            # add backward edge
            backward_data = copy.deepcopy(data)
            backward_data['eid'] = avail_eid
            avail_eid += 1
            backward_data['coords'].reverse()
            g.add_edge(v, u, **backward_data)
            edge_spatial_idx.insert(backward_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
            edge_idx[backward_data['eid']] = (v, u)
        print('# of nodes:{}'.format(g.number_of_nodes()))
        print('# of edges:{}'.format(g.number_of_edges()))
        return RoadNetwork(g, edge_spatial_idx, edge_idx)

    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return [self.edge_idx[eid] for eid in eids]

    def remove_edge(self, u, v):
        edge_data = self[u][v]
        coords = edge_data['coords']
        mbr = MBR.cal_mbr(coords)
        # delete self.edge_idx[eifrom edge index
        del self.edge_idx[edge_data['eid']]
        # delete from spatial index
        self.edge_spatial_idx.delete(edge_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # delete from graph
        super(UndirRoadNetwork, self).remove_edge(u, v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        coords = attr['coords']
        mbr = MBR.cal_mbr(coords)
        attr['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        # add edge to edge index
        self.edge_idx[attr['eid']] = (u_of_edge, v_of_edge)
        # add edge to spatial index
        self.edge_spatial_idx.insert(attr['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # add edge to graph
        super(UndirRoadNetwork, self).add_edge(u_of_edge, v_of_edge, **attr)


class RoadNetwork(nx.DiGraph):
    def __init__(self, g, edge_spatial_idx, edge_idx):
        super(RoadNetwork, self).__init__(g)
        # entry: eid
        self.edge_spatial_idx = edge_spatial_idx
        # eid -> edge key (start_coord, end_coord)
        self.edge_idx = edge_idx

    def range_query(self, mbr):
        """
        spatial range query
        :param mbr: query mbr
        :return: qualified edge keys
        """
        eids = self.edge_spatial_idx.intersection((mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        return [self.edge_idx[eid] for eid in eids]

    def remove_edge(self, u, v):
        edge_data = self[u][v]
        coords = edge_data['coords']
        mbr = MBR.cal_mbr(coords)
        # delete self.edge_idx[eifrom edge index
        del self.edge_idx[edge_data['eid']]
        # delete from spatial index
        self.edge_spatial_idx.delete(edge_data['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # delete from graph
        super(RoadNetwork, self).remove_edge(u, v)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        coords = attr['coords']
        mbr = MBR.cal_mbr(coords)
        attr['length'] = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        # add edge to edge index
        self.edge_idx[attr['eid']] = (u_of_edge, v_of_edge)
        # add edge to spatial index
        self.edge_spatial_idx.insert(attr['eid'], (mbr.min_lng, mbr.min_lat, mbr.max_lng, mbr.max_lat))
        # add edge to graph
        super(RoadNetwork, self).add_edge(u_of_edge, v_of_edge, **attr)


def _shp_path(path, layer_name):
    """接受目录或具体 .shp 路径，返回 ogr 能打开的 .shp 路径。"""
    if os.path.isdir(path):
        return os.path.join(path, layer_name + '.shp')
    return path


def load_rn_shp(path, is_directed=True):
    """从 shapefile 读路网，不依赖 nx.read_shp (nx>=3 已移除)。

    节点 key = (lng, lat)；边 attr: eid, length, coords(SPoint[])。
    与原实现保持一致，返回 RoadNetwork(DiGraph) 或 UndirRoadNetwork(Graph)。
    path 可以是目录 (含 edges.shp) 或 edges.shp 本身。
    """
    edge_spatial_idx = Rtree()
    edge_idx = {}
    if is_directed:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    shp_file = _shp_path(path, 'edges')
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_file, 0)
    if ds is None:
        raise FileNotFoundError('cannot open shapefile: {}'.format(shp_file))
    layer = ds.GetLayer()
    eid = 0
    for feat in layer:
        geom = feat.GetGeometryRef()
        if geom is None or ogr.GT_Flatten(geom.GetGeometryType()) != ogr.wkbLineString:
            continue
        n_pts = geom.GetPointCount()
        if n_pts < 2:
            continue
        coords = []
        for i in range(n_pts):
            lng, lat = geom.GetX(i), geom.GetY(i)
            coords.append(SPoint(lat, lng))
        u = (coords[0].lng, coords[0].lat)
        v = (coords[-1].lng, coords[-1].lat)
        length = sum([distance(coords[i], coords[i + 1]) for i in range(len(coords) - 1)])
        g.add_node(u, pt=SPoint(u[1], u[0]))
        g.add_node(v, pt=SPoint(v[1], v[0]))
        edge_data = {'eid': eid, 'length': length, 'coords': coords}
        g.add_edge(u, v, **edge_data)
        env = geom.GetEnvelope()  # (minX, maxX, minY, maxY)
        edge_spatial_idx.insert(eid, (env[0], env[2], env[1], env[3]))
        edge_idx[eid] = (u, v)
        eid += 1
    ds = None
    print('# of nodes:{}'.format(g.number_of_nodes()))
    print('# of edges:{}'.format(g.number_of_edges()))
    if not is_directed:
        return UndirRoadNetwork(g, edge_spatial_idx, edge_idx)
    else:
        return RoadNetwork(g, edge_spatial_idx, edge_idx)


def store_rn_shp(rn, target_path):
    """把路网写成 shapefile (edges.shp + nodes.shp)，不依赖 nx.write_shp。

    无向图会先 to_directed()，每条边写一条 (原 forward 方向)。
    target_path 为目录；坐标 WGS84 (lng, lat)。
    """
    print('# of nodes:{}'.format(rn.number_of_nodes()))
    print('# of edges:{}'.format(rn.number_of_edges()))
    if not rn.is_directed():
        rn = rn.to_directed()

    os.makedirs(target_path, exist_ok=True)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    # --- edges.shp ---
    drv = ogr.GetDriverByName('ESRI Shapefile')
    edges_path = os.path.join(target_path, 'edges.shp')
    if os.path.exists(edges_path):
        drv.DeleteDataSource(edges_path)
    ds = drv.CreateDataSource(edges_path)
    layer = ds.CreateLayer('edges', srs=srs, geom_type=ogr.wkbLineString)
    layer.CreateField(ogr.FieldDefn('eid', ogr.OFTInteger))
    layer.CreateField(ogr.FieldDefn('length', ogr.OFTReal))
    for u, v, data in rn.edges(data=True):
        coords = data['coords']
        geom = ogr.Geometry(ogr.wkbLineString)
        for coord in coords:
            geom.AddPoint(coord.lng, coord.lat)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(geom)
        feat.SetField('eid', data.get('eid', 0))
        feat.SetField('length', data.get('length', 0.0))
        layer.CreateFeature(feat)
        feat = None
    ds = None

    # --- nodes.shp ---
    nodes_path = os.path.join(target_path, 'nodes.shp')
    if os.path.exists(nodes_path):
        drv.DeleteDataSource(nodes_path)
    ds = drv.CreateDataSource(nodes_path)
    layer = ds.CreateLayer('nodes', srs=srs, geom_type=ogr.wkbPoint)
    for n, data in rn.nodes(data=True):
        geom = ogr.Geometry(ogr.wkbPoint)
        geom.AddPoint(n[0], n[1])  # (lng, lat)
        feat = ogr.Feature(layer.GetLayerDefn())
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None
    ds = None
