import os
import csv
import argparse
from tqdm import tqdm
from datetime import datetime

def parse_header(line):
    """
    解析 Header 行
    Input: #,tid,oid,start_time,end_time,length
    Example: #,1db...62_2018...04_2018...56,1db...62,2018/09/30 12:57:04,2018/09/30 13:13:56,9.0676 km
    Output: dict {tid, oid, start_time, end_time}
    """
    parts = line.strip().split(',')
    if len(parts) < 2:
        return None
    
    # parts[0] is '#'
    return {
        'tid': parts[1],
        'oid': parts[2],
        'start_time_str': parts[3], # 保持原始字符串，或者统一格式化
        'end_time_str': parts[4]
    }

def parse_data_line(line):
    """
    解析数据行 (坐标点)
    Input: 2018/09/30 16:43:57,34.2709257,108.9845113,13861,34.2709128,108.9845111,1.44,556.55
    Output: dict or None
    """
    parts = line.strip().split(',')
    if len(parts) < 6:
        return None
    
    # 检查 EdgeID (第4列, index 3) 是否有效
    edge_id_str = parts[3].strip()
    if edge_id_str == 'None' or edge_id_str == '':
        return None # 丢弃未匹配点
    
    try:
        # 提取匹配后的经纬度 (Lat: index 4, Lon: index 5)
        # 注意 CSV WKT 需要 "Lon Lat" 顺序
        matched_lat = float(parts[4])
        matched_lon = float(parts[5])
        
        return f"{matched_lon} {matched_lat}"
    except Exception:
        return None

def process_file(input_path, output_path):
    """
    处理包含多条轨迹的单个文件
    """
    rows = []
    
    # 当前正在处理的轨迹缓存
    current_meta = None
    current_points = []
    
    def save_current_traj():
        """闭包函数：保存当前缓存的轨迹到 rows"""
        if current_meta and len(current_points) >= 2:
            # 构建 WKT
            wkt = f"LINESTRING ({', '.join(current_points)})"
            
            # 格式化时间 (尝试将斜杠转为横杠，保持 ISO 风格)
            # Input: 2018/09/30 12:57:04 -> Output: 2018-09-30 12:57:04
            s_time = current_meta['start_time_str'].replace('/', '-')
            e_time = current_meta['end_time_str'].replace('/', '-')
            
            rows.append([
                current_meta['tid'],
                current_meta['oid'],
                s_time,
                e_time,
                wkt
            ])

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # === Case A: Header 行 (新轨迹开始) ===
            if line.startswith('#'):
                # 1. 如果之前有正在处理的轨迹，先保存它
                save_current_traj()
                
                # 2. 解析新的 Header
                current_meta = parse_header(line)
                current_points = [] # 重置点列表
                
            # === Case B: 数据行 (坐标点) ===
            else:
                # 只有当 header 已经解析成功后，才收集点
                if current_meta:
                    pt_str = parse_data_line(line)
                    if pt_str:
                        current_points.append(pt_str)
    
    # 循环结束后，别忘了保存最后一条轨迹
    save_current_traj()
    
    # 写入 CSV
    if rows:
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['tid', 'oid', 'start_time', 'end_time', 'wkt'])
            writer.writerows(rows)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]
    print(f"[INFO] Found {len(files)} trajectory files.")

    for filename in tqdm(files):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace(".txt", ".csv")
        output_path = os.path.join(output_folder, output_filename)

        try:
            process_file(input_path, output_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {filename} : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, default=r"./xian/seg_mm_traj")
    parser.add_argument("-o", "--output", required=True, default=r"./xian/seg_mm_traj_csv")
    args = parser.parse_args()
    
    process_folder(args.input, args.output)
    print("\n[INFO] Done.")
