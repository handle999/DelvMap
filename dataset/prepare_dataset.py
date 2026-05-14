"""
DelvMap 数据制备脚本
将原始 sat_img.png + traj_heat.png 转换为 create_dataset.py 需要的格式

输出结构（每个文件夹一张完整大图，create_dataset.py 会从中裁剪 patch）：
data_root/
├── basemap/         # 现有道路 (highway, 二值化 0/255)
├── src/             # 卫星影像 (RGB)
├── traj/            # 轨迹路径 (二值化 0/255)
├── trajpoint/       # 轨迹点 (二值化 0/255)
├── building_label/ # 建筑 (building, 二值化 0/255)
└── map_label/
    └── label_width2/ # 真值道路 (完整道路)

依赖: pip install numpy opencv-python tqdm osmium
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
import osmium


# ==========================================
# 配置
# ==========================================
CONFIG = {
    # 地理范围 (来自 RUN.md)
    'lat_min': 34.206385,
    'lat_max': 34.279658,
    'lon_min': 108.917423,
    'lon_max': 108.99286,

    # 图像尺寸 (来自 RUN.md)
    'img_w': 5625,
    'img_h': 6610,

    # OSM PBF 文件
    'osm_pbf': r"e:\School\2025\20250311Road\GraphBased\DelvMap\dataset\osm\xian-plus-190101-multi.osm.pbf",

    # 道路宽度 (像素)
    'road_width': 2,

    # 路径
    'rawdata_dir': r"e:\School\2025\20250311Road\GraphBased\DelvMap\rawdata",
    'output_dir': r"e:\School\2025\20250311Road\GraphBased\DelvMap\dataset\delvmap_data_1919",
}


# ==========================================
# 工具函数
# ==========================================
# def geo_to_pixel(lat, lon, img_w, img_h, config):
#     """地理坐标 -> 像素坐标"""
#     x = int((lon - config['lon_min']) / (config['lon_max'] - config['lon_min']) * img_w)
#     y = int((config['lat_max'] - lat) / (config['lat_max'] - config['lat_min']) * img_h)
#     # Y轴翻转 (地理坐标向上，图像坐标向下)
#     y = img_h - 1 - y
#     return x, y
import math

def wgs84_to_mercator(lon, lat):
    """将 WGS84 经纬度转换为 Web Mercator (EPSG:3857) 平面坐标"""
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y

def geo_to_pixel(lat, lon, img_w, img_h, config):
    """(修复版) 墨卡托投影下的坐标 -> 像素坐标"""
    # 1. 获取图片边界的墨卡托坐标
    x_min, y_min = wgs84_to_mercator(config['lon_min'], config['lat_min'])
    x_max, y_max = wgs84_to_mercator(config['lon_max'], config['lat_max'])
    
    # 2. 将当前点转为墨卡托坐标
    x_m, y_m = wgs84_to_mercator(lon, lat)
    
    # 3. 在墨卡托平面上进行线性插值计算像素
    x = int((x_m - x_min) / (x_max - x_min) * img_w)
    
    # 注意：墨卡托坐标的 Y 轴是向北增加的，而图像像素的 Y 轴是向南(向下)增加的
    y = int((y_max - y_m) / (y_max - y_min) * img_h)
    
    return x, y


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def binaryzation(img):
    """二值化：>0 -> 255, 0 -> 0"""
    img_binary = np.where(img > 0, 255, 0).astype(np.uint8)
    return img_binary


# ==========================================
# 1. OSM 数据加载器
# ==========================================
class OSMExtractor(osmium.SimpleHandler):
    """从 OSM PBF 提取道路和建筑"""

    def __init__(self):
        super().__init__()
        self.nodes = {}           # node_id -> (lat, lon)
        self.roads = []           # 道路 (node_refs)
        self.buildings = []       # 建筑 (node_refs)

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        tags = dict(w.tags)

        # 提取道路 (highway)
        if 'highway' in tags:
            highway = tags['highway']
            # 排除不想要的道路类型
            exclude = ['footway', 'path', 'service', 'parking', 'driveway',
                       'cycleway', 'steps', 'bridleway']
            if highway not in exclude and len(w.nodes) >= 2:
                self.roads.append([n.ref for n in w.nodes])

        # 提取建筑 (building)
        if 'building' in tags and len(w.nodes) >= 3:
            self.buildings.append([n.ref for n in w.nodes])


def load_osm_data(pbf_path):
    """加载 OSM 数据"""
    print(f"正在解析 OSM PBF: {pbf_path}")
    handler = OSMExtractor()
    handler.apply_file(pbf_path)

    print(f"  节点数: {len(handler.nodes)}")
    print(f"  道路数: {len(handler.roads)}")
    print(f"  建筑数: {len(handler.buildings)}")

    return handler


# ==========================================
# 2. 渲染函数
# ==========================================
def render_roads(roads, nodes, img_w, img_h, config):
    """渲染道路为二值图像"""
    print(f"正在渲染道路 ({img_w}x{img_h})...")
    road_img = np.zeros((img_h, img_w), dtype=np.uint8)

    # 将 node_refs 转换为坐标序列
    valid_roads = []
    for node_refs in roads:
        coords = []
        valid = True
        for ref in node_refs:
            if ref in nodes:
                coords.append(nodes[ref])
            else:
                valid = False
                break
        if valid and len(coords) >= 2:
            valid_roads.append(coords)

    print(f"  有效道路: {len(valid_roads)} 条")

    for coords in tqdm(valid_roads):
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[i+1]
            x1, y1 = geo_to_pixel(lat1, lon1, img_w, img_h, config)
            x2, y2 = geo_to_pixel(lat2, lon2, img_w, img_h, config)

            # 裁剪到图像范围
            if (0 <= x1 < img_w and 0 <= y1 < img_h) or (0 <= x2 < img_w and 0 <= y2 < img_h):
                cv2.line(road_img, (x1, y1), (x2, y2), 255, thickness=config['road_width'])

    # 二值化
    road_img = binaryzation(road_img)
    return road_img


def render_buildings(buildings, nodes, img_w, img_h, config):
    """渲染建筑为二值图像"""
    print(f"正在渲染建筑 ({img_w}x{img_h})...")
    building_img = np.zeros((img_h, img_w), dtype=np.uint8)

    # 构建有效建筑列表
    valid_buildings = []
    for node_refs in buildings:
        poly = []
        valid = True
        for ref in node_refs:
            if ref in nodes:
                poly.append(nodes[ref])
            else:
                valid = False
                break
        if valid and len(poly) >= 3:
            valid_buildings.append(poly)

    print(f"  有效建筑: {len(valid_buildings)} 个")

    for poly in tqdm(valid_buildings):
        poly_pixels = []
        for (lat, lon) in poly:
            x, y = geo_to_pixel(lat, lon, img_w, img_h, config)
            if 0 <= x < img_w and 0 <= y < img_h:
                poly_pixels.append([x, y])

        if len(poly_pixels) >= 3:
            pts = np.array(poly_pixels, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(building_img, [pts], (255,))

    # 二值化
    building_img = binaryzation(building_img)
    return building_img


# ==========================================
# 3. 主函数
# ==========================================
def main():
    config = CONFIG
    img_w = config['img_w']
    img_h = config['img_h']
    output_dir = config['output_dir']

    # 创建输出目录结构
    subdirs = ['basemap', 'src', 'traj', 'trajpoint', 'building_label', 'map_label/label_width2']
    for sub in subdirs:
        ensure_dir(os.path.join(output_dir, sub))

    # ========== 1. 处理轨迹数据 ==========
    print("\n=== 处理轨迹数据 ===")
    traj_path = os.path.join(config['rawdata_dir'], 'traj_heat.png')
    traj_heat = cv2.imread(traj_path, cv2.IMREAD_GRAYSCALE)

    if traj_heat is None:
        raise FileNotFoundError(f"找不到轨迹文件: {traj_path}")

    # 调整尺寸
    if traj_heat.shape[:2] != (img_h, img_w):
        print(f"  调整尺寸: {traj_heat.shape[1]}x{traj_heat.shape[0]} -> {img_w}x{img_h}")
        traj_heat = cv2.resize(traj_heat, (img_w, img_h))

    # 二值化
    traj_heat = binaryzation(traj_heat)

    # trajpoint = 轨迹点
    trajpoint_path = os.path.join(output_dir, 'trajpoint', '0.png')
    cv2.imwrite(trajpoint_path, traj_heat)
    print(f"  保存: trajpoint/0.png")

    # traj = 轨迹路径 (闭运算连接点)
    kernel = np.ones((3, 3), np.uint8)
    traj = cv2.morphologyEx(traj_heat, cv2.MORPH_CLOSE, kernel)
    traj_path_out = os.path.join(output_dir, 'traj', '0.png')
    cv2.imwrite(traj_path_out, traj)
    print(f"  保存: traj/0.png")

    # ========== 2. 处理卫星影像 ==========
    print("\n=== 处理卫星影像 ===")
    sat_path = os.path.join(config['rawdata_dir'], 'sat_img.png')
    sat = cv2.imread(sat_path)

    if sat is None:
        raise FileNotFoundError(f"找不到卫星影像: {sat_path}")

    # 调整尺寸
    if sat.shape[:2] != (img_h, img_w):
        print(f"  调整尺寸: {sat.shape[1]}x{sat.shape[0]} -> {img_w}x{img_h}")
        sat = cv2.resize(sat, (img_w, img_h))

    src_path = os.path.join(output_dir, 'src', '0.png')
    cv2.imwrite(src_path, sat)
    print(f"  保存: src/0.png")

    # ========== 3. 加载 OSM 数据 ==========
    print("\n=== 加载 OSM 数据 ===")
    handler = load_osm_data(config['osm_pbf'])

    # ========== 4. 生成 basemap (现有道路) ==========
    print("\n=== 生成 basemap (现有道路) ===")
    basemap = render_roads(handler.roads, handler.nodes, img_w, img_h, config)
    basemap_path = os.path.join(output_dir, 'basemap', '0.png')
    cv2.imwrite(basemap_path, basemap)
    print(f"  保存: basemap/0.png")

    # ========== 5. 生成 building_label ==========
    print("\n=== 生成 building_label ===")
    buildings = render_buildings(handler.buildings, handler.nodes, img_w, img_h, config)
    building_path = os.path.join(output_dir, 'building_label', '0.png')
    cv2.imwrite(building_path, buildings)
    print(f"  保存: building_label/0.png")

    # ========== 6. 生成 map_label (真值道路) ==========
    # 这里 basemap 就是完整路网，如果要模拟缺失路网，可以手动删除一些道路
    label_path = os.path.join(output_dir, 'map_label/label_width2', '0.png')
    cv2.imwrite(label_path, basemap)  # 用相同的数据作为真值
    print(f"  保存: map_label/label_width2/0.png")

    print("\n" + "="*50)
    print("数据制备完成!")
    print(f"输出目录: {output_dir}")
    print("="*50)
    print("\n生成的文件:")
    print("  basemap/0.png          - 现有道路 (highway, 二值化)")
    print("  src/0.png              - 卫星影像")
    print("  traj/0.png             - 轨迹路径")
    print("  trajpoint/0.png        - 轨迹点")
    print("  building_label/0.png  - 建筑 (building, 二值化)")
    print("  map_label/0.png        - 真值道路")
    print("\n下一步: 运行 create_dataset.py 进行裁剪")


if __name__ == '__main__':
    main()