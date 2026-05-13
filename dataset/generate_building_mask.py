import os
import json
import math
import argparse
import numpy as np
import cv2
import osmium
from rtree import index

# ==========================================
# 1. 纯 Python 字典策略的解析器 (模仿 download_use_pbf)
# ==========================================
class BuildingHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        # 抛弃 C++ 缓存，回归朴素的 Python 字典
        self.nodes = {}      # node_id -> (lat, lon)
        self.ways = []       # list of node_id sequences

    def node(self, n):
        # 简单粗暴地存入字典
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        # 只提取建筑物的 ID (n.ref)，绝对不碰 n.location！
        if 'building' in w.tags and len(w.nodes) >= 3:
            self.ways.append([n.ref for n in w.nodes])

def norm(x, nd=7): 
    return round(x, nd)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_file", required=True, help="Path to .osm.pbf file")
    parser.add_argument("--configs", nargs="+", help="Dataset JSON config files")
    args = parser.parse_args()

    # --- Step 1: 解析 PBF 文件，提取到内存 ---
    print(f"[INFO] 正在解析 PBF 到 Python 内存字典: {args.map_file} ...")
    handler = BuildingHandler()
    # 注意：我们直接去掉 locations=True，彻底关闭引发崩溃的 C++ 内存映射！
    handler.apply_file(args.map_file)
    print(f"[INFO] PBF 读取完毕！共读取了 {len(handler.nodes)} 个点，{len(handler.ways)} 个建筑物轮廓。")

    # --- Step 2: 在纯 Python 环境下安全构建 R-Tree ---
    print("[INFO] 正在组装坐标并构建 R-Tree 空间索引 ...")
    spatial_idx = index.Index()
    building_id = 0
    
    for way_refs in handler.ways:
        poly = []
        valid = True
        # 手动映射查字典：Node ID -> Lat, Lon
        for ref in way_refs:
            if ref in handler.nodes:
                poly.append(handler.nodes[ref])
            else:
                valid = False
                break
                
        if not valid or len(poly) < 3:
            continue
            
        # 构建 BBox: (min_lon, min_lat, max_lon, max_lat)
        lats = [p[0] for p in poly]
        lons = [p[1] for p in poly]
        bbox = (min(lons), min(lats), max(lons), max(lats))
        
        # 安全地将多边形插入 R-Tree
        spatial_idx.insert(building_id, bbox, obj=poly)
        building_id += 1
        
    print(f"[INFO] R-Tree 构建完成！有效建筑物数量: {building_id}。")
    
    # 为了节省内存，构建完 R-Tree 后可以清空原始字典
    del handler.nodes
    del handler.ways

    # --- Step 3: 加载 Configs ---
    dataset_cfg = []
    total_regions = 0 
    for name_cfg in args.configs:
        dataset_cfg_ = json.load(open(name_cfg, "r"))
        for item in dataset_cfg_:
            size = item["size"]
            lat_min, lon_min = item["lat_min"], item["lon_min"]
            lat_max, lon_max = item["lat_max"], item["lon_max"]
            
            dlat = size / 111111.0           
            dlon = size / (111111.0 * math.cos(math.radians(lat_min)))  
            lat_n = math.ceil((lat_max - lat_min) / dlat)  
            lon_n = math.ceil((lon_max - lon_min) / dlon)  
            
            dataset_cfg.append({
                "lat": lat_min, "lon": lon_min, "lat_n": lat_n, "lon_n": lon_n, 
                "size": size, "region_name": item["region"], "year": item["year"]
            })
            total_regions += lat_n * lon_n   

    # --- Step 4: R-Tree 极速查询与 OpenCV 渲染 ---
    c = 0
    for item in dataset_cfg:
        ilat, ilon = item["lat_n"], item["lon_n"]
        lat_origin, lon_origin = item["lat"], item["lon"]
        size = item["size"]
        
        dataset_folder = f"{item['region_name']}_{item['year']}_{size}"
        os.makedirs(dataset_folder, exist_ok=True)

        for i in range(ilat):
            for j in range(ilon):
                lat_st = norm(lat_origin + size/111111.0 * i)
                lon_st = norm(lon_origin + size/111111.0 * j / math.cos(math.radians(lat_origin)))
                lat_ed = norm(lat_origin + size/111111.0 * (i+1))
                lon_ed = norm(lon_origin + size/111111.0 * (j+1) / math.cos(math.radians(lat_origin)))
                
                query_bbox = (lon_st, lat_st, lon_ed, lat_ed)
                mask_img = np.zeros((size, size), dtype=np.uint8)
                
                # R-Tree 空间查询
                intersecting_buildings = list(spatial_idx.intersection(query_bbox, objects=True))
                
                for item_obj in intersecting_buildings:
                    poly = item_obj.object
                    poly_pixels = []
                    for (lat, lon) in poly:
                        x = (lon - lon_st) / (lon_ed - lon_st) * size
                        y = (lat_ed - lat) / (lat_ed - lat_st) * size
                        poly_pixels.append([x, y])
                    
                    pts = np.array(poly_pixels, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask_img, [pts], color=(255,))

                output_path = os.path.join(dataset_folder, f"region_{c}_building.png")
                cv2.imwrite(output_path, mask_img)
                
                if c % 100 == 0:
                    print(f"Processed {c}/{total_regions} -> {output_path}")
                c += 1

# run
# python generate_building_mask.py --map_file=./osm/xian-plus-190101.osm.pbf --config=./config/xian.json