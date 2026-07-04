import os
import csv
import math
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm
from datetime import datetime
import re

# ==========================================
# 辅助工具：Haversine 距离计算
# ==========================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的 Haversine 距离（单位：米）
    """
    R = 6371000  # 地球半径 (米)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# ==========================================
# 辅助类：Trajectory (仿照 parse_traj_file 的返回对象)
# ==========================================
class CsvTrajectory:
    def __init__(self, oid, pt_list, start_time, end_time):
        self.oid = oid
        self.pt_list = pt_list  # [(lon, lat), ...]
        self.start_time = start_time
        self.end_time = end_time
        self._length = None

    def get_length(self):
        """计算轨迹总长度 (米)"""
        if self._length is not None:
            return self._length
        
        dist = 0.0
        for i in range(len(self.pt_list) - 1):
            p1 = self.pt_list[i]
            p2 = self.pt_list[i+1]
            # 注意：pt_list 存的是 (lon, lat)，计算需传 (lat, lon)
            dist += haversine_distance(p1[1], p1[0], p2[1], p2[0])
        self._length = dist
        return dist

    def get_duration(self):
        """计算持续时间 (秒)"""
        delta = self.end_time - self.start_time
        return delta.total_seconds()

    def get_time_interval(self):
        """计算平均时间间隔 (秒)"""
        # 由于CSV只存储了起止时间，这里假设采样是均匀的
        # Time Interval = Total Duration / (Num Segments)
        if len(self.pt_list) <= 1:
            return 0
        return self.get_duration() / (len(self.pt_list) - 1)

    def get_distance_interval(self):
        """计算平均距离间隔 (米)"""
        if len(self.pt_list) <= 1:
            return 0
        return self.get_length() / (len(self.pt_list) - 1)

# ==========================================
# 解析函数
# ==========================================
def parse_csv_file(filepath):
    """
    解析单个 CSV 文件，返回 CsvTrajectory 对象列表
    CSV 格式: tid,oid,start_time,end_time,wkt
    """
    trajs = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                oid = row['oid']
                start_str = row['start_time']
                end_str = row['end_time']
                wkt = row['wkt']

                # 解析时间
                try:
                    t_start = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
                    t_end = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue

                # 解析 WKT: LINESTRING (lon lat, lon lat, ...)
                # 使用简单的字符串处理提取坐标
                coords_str = wkt.replace('LINESTRING (', '').replace(')', '')
                pt_list = []
                if coords_str.strip():
                    points = coords_str.split(', ')
                    for p in points:
                        lon, lat = map(float, p.split(' '))
                        pt_list.append((lon, lat))
                
                # 创建对象
                if len(pt_list) > 1:
                    trajs.append(CsvTrajectory(oid, pt_list, t_start, t_end))
                    
    except Exception as e:
        print(f"[WARN] Error parsing {filepath}: {e}")
        
    return trajs

# ==========================================
# 统计主函数 (仿照 statistics.py)
# ==========================================
def statistics(traj_dir):
    # 构造输出目录：path/to/data_csv -> path/to/data_csv_stat/
    # 处理路径末尾可能的斜杠
    traj_dir = os.path.normpath(traj_dir)
    parent_dir = os.path.dirname(traj_dir)
    base_name = os.path.basename(traj_dir)
    
    stat_dirname = base_name + '_stat'
    stat_dir = os.path.join(parent_dir, stat_dirname)
    
    os.makedirs(stat_dir, exist_ok=True)
    print(f"[INFO] Statistics will be saved to: {stat_dir}")

    length_data = []
    duration_data = []
    seq_len_data = []
    traj_avg_time_interval_data = []
    traj_avg_dist_interval_data = []
    oids = set()
    tot_pts = 0
    tot_trajs = 0
    stats = {}

    # 遍历 CSV 文件
    files = [f for f in os.listdir(traj_dir) if f.endswith('.csv')]
    
    for filename in tqdm(files, desc="Calculating Statistics"):
        trajs = parse_csv_file(os.path.join(traj_dir, filename))
        tot_trajs += len(trajs)
        
        for traj in trajs:
            # 过滤单点轨迹 (虽然 parse_csv_file 已过滤，双重保险)
            if len(traj.pt_list) <= 1: 
                continue
            
            oids.add(traj.oid)
            nb_pts = len(traj.pt_list)
            tot_pts += nb_pts
            
            seq_len_data.append(nb_pts)
            length_data.append(traj.get_length() / 1000.0)  # 米 -> 千米
            duration_data.append(traj.get_duration() / 60.0) # 秒 -> 分钟
            traj_avg_time_interval_data.append(traj.get_time_interval())
            traj_avg_dist_interval_data.append(traj.get_distance_interval())

    print('#objects:{}'.format(len(oids)))
    print('#points:{}'.format(tot_pts))
    print('#trajectories:{}'.format(tot_trajs))
    
    stats['#objects'] = len(oids)
    stats['#points'] = tot_pts
    stats['#trajectories'] = tot_trajs
    
    with open(os.path.join(stat_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

    # 绘图逻辑
    # 1. Number of Points Distribution
    if seq_len_data:
        plt.hist(seq_len_data, weights=np.ones(len(seq_len_data)) / len(seq_len_data), bins=50, alpha=0.7, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('#Points')
        plt.ylabel('Percentage')
        plt.title('Distribution of Points per Trajectory')
        plt.savefig(os.path.join(stat_dir, 'nb_points_dist.png'))
        plt.clf()

    # 2. Length Distribution
    if length_data:
        plt.hist(length_data, weights=np.ones(len(length_data)) / len(length_data), bins=50, alpha=0.7, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Length (KM)')
        plt.ylabel('Percentage')
        plt.title('Distribution of Trajectory Lengths')
        plt.savefig(os.path.join(stat_dir, 'length_dist.png'))
        plt.clf()

    # 3. Duration Distribution
    if duration_data:
        plt.hist(duration_data, weights=np.ones(len(duration_data)) / len(duration_data), bins=50, alpha=0.7, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Duration (Min)')
        plt.ylabel('Percentage')
        plt.title('Distribution of Trajectory Durations')
        plt.savefig(os.path.join(stat_dir, 'duration_dist.png'))
        plt.clf()

    # 4. Time Interval Distribution
    if traj_avg_time_interval_data:
        plt.hist(traj_avg_time_interval_data,
                 weights=np.ones(len(traj_avg_time_interval_data)) / len(traj_avg_time_interval_data), bins=50, alpha=0.7, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Time Interval (Sec)')
        plt.ylabel('Percentage')
        plt.title('Distribution of Avg Time Intervals')
        plt.savefig(os.path.join(stat_dir, 'time_interval_dist.png'))
        plt.clf()

    # 5. Distance Interval Distribution
    if traj_avg_dist_interval_data:
        plt.hist(traj_avg_dist_interval_data,
                 weights=np.ones(len(traj_avg_dist_interval_data)) / len(traj_avg_dist_interval_data), bins=50, alpha=0.7, edgecolor='black')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xlabel('Distance Interval (Meter)')
        plt.ylabel('Percentage')
        plt.title('Distribution of Avg Distance Intervals')
        plt.savefig(os.path.join(stat_dir, 'distance_interval_dist.png'))
        plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate statistics for CSV trajectory files")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing .csv trajectory files")
    args = parser.parse_args()
    
    statistics(args.input)
    