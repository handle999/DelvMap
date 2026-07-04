import os
import csv
from tqdm import tqdm
import argparse
from osgeo import ogr
from .path import parse_path_file 


def load_rn_geometries(shp_path):
    """
    读取路网 Shapefile，建立 eid -> [(lng, lat), (lng, lat)...] 的映射字典
    """
    print(f"[INFO] Loading road network from {shp_path}...")
    eid_to_points = {}
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_path, 0) # 0 means read-only
    if data_source is None:
        raise FileNotFoundError(f"Cannot open Shapefile: {shp_path}")
        
    layer = data_source.GetLayer()
    
    for feature in layer:
        eid = feature.GetField("eid")       # 直接读取小写的 eid 字段
        geom = feature.GetGeometryRef()
        if geom is not None:
            eid_to_points[int(eid)] = geom.GetPoints()
            
    print(f"[INFO] Loaded {len(eid_to_points)} edges with geometries.")
    return eid_to_points


def points_to_wkt(points):
    """
    将点集转化为 WKT LINESTRING 格式
    """
    if not points or len(points) < 2:
        return "LINESTRING EMPTY"
    # 提取 (lng, lat) 忽略可能存在的 Z 值，并保留7位小数
    pt_strs = [f"{p[0]:.7f} {p[1]:.7f}" for p in points]
    return "LINESTRING (" + ", ".join(pt_strs) + ")"


def process_folder(input_folder, output_folder, rn_path):
    """
    将 path 的 txt 文件转换为带 WKT 的 csv 文件
    """
    # 1. 预加载路网几何信息
    eid_to_points = load_rn_geometries(rn_path)

    # 2. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 3. 遍历输入文件夹
    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    
    for filename in tqdm(files, desc="Converting Paths to CSV"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace(".txt", ".csv")
        output_path = os.path.join(output_folder, output_filename)

        try:
            # 4. 解析 path 文本
            paths = parse_path_file(input_path)

            # 5. 写入 CSV 文件
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(['tid', 'oid', 'start_time', 'end_time', 'wkt'])
                
                for path in paths:
                    if not path.path_entities:
                        continue
                        
                    # 组装轨迹点
                    all_points = []
                    for entity in path.path_entities:
                        eid = entity.eid
                        if eid in eid_to_points:
                            pts = eid_to_points[eid]
                            if not all_points:
                                all_points.extend(pts)
                            else:
                                # 为避免首尾连接处坐标重复，如果当前路段起点与上个路段终点相同，则跳过第一个点
                                if all_points[-1][0] == pts[0][0] and all_points[-1][1] == pts[0][1]:
                                    all_points.extend(pts[1:])
                                else:
                                    all_points.extend(pts)
                    
                    # 生成 WKT
                    wkt_str = points_to_wkt(all_points)
                    
                    # 只有凑齐了可以连成线的坐标才写入
                    if wkt_str != "LINESTRING EMPTY":
                        writer.writerow([
                            path.pid,  # 用 pid 作为 tid
                            path.oid,
                            path.path_entities[0].enter_time.strftime('%Y-%m-%d %H:%M:%S'),
                            path.path_entities[-1].leave_time.strftime('%Y-%m-%d %H:%M:%S'),
                            wkt_str
                        ])
        except Exception as e:
            print(f"\n[ERROR] Failed to process {filename} : {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert path txt files to CSV with WKT Geometries (QGIS compatible)"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input folder containing path .txt files (e.g., ../xian/seg_mm_path_5)"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output folder for converted .csv files (e.g., ../xian/seg_mm_path_5_csv)"
    )
    
    parser.add_argument(
        "--rn_path",
        required=True,
        help="Path to the road network shapefile (e.g., ../xian/osm/rn-comp-xa-190101-didi/edges.shp)"
    )
    
    args = parser.parse_args()
    print(args)

    process_folder(args.input, args.output, args.rn_path)
    print("\n[INFO] All files processed successfully.")
