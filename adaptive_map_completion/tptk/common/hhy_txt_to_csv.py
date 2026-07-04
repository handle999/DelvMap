import os
import csv
from tqdm import tqdm
import argparse
from .trajectory import parse_traj_file

def process_folder(input_folder, output_folder):
    """
    Convert trajectory txt files to csv files (QGIS friendly)

    Args:
        input_folder (Path): folder containing .txt trajectory files
        output_folder (Path): folder to save converted .csv files
    """
    # 1. 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 2. 遍历输入文件夹
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_filename = filename.replace(".txt", ".csv")
            output_path = os.path.join(output_folder, output_filename)

            # print(f"正在转换: {filename} -> {output_filename}")

            try:
                # 3. 调用 trajectory.py 中的解析函数. 注意：由于你的数据包含路段ID等信息，必须指定 traj_type='mm'
                trajs = parse_traj_file(input_path, traj_type='mm')

                # 4. 写入 CSV 文件
                with open(output_path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    # 写入表头：轨迹ID，对象ID，开始时间，结束时间，WKT几何体
                    writer.writerow(['tid', 'oid', 'start_time', 'end_time', 'wkt'])
                    
                    for traj in trajs:
                        writer.writerow([
                            traj.tid,
                            traj.oid,
                            traj.get_start_time().strftime('%Y-%m-%d %H:%M:%S'),
                            traj.get_end_time().strftime('%Y-%m-%d %H:%M:%S'),
                            traj.to_wkt()
                        ])
            except Exception as e:
                print(f"[ERROR] Failed to process {filename} : {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert trajectory txt files to CSV (QGIS compatible)"
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        default=r"./xian/seg_mm_traj",
        help="Input folder containing trajectory .txt files"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        default=r"./xian/seg_mm_traj_csv",
        help="Output folder for converted .csv files"
    )
    
    args = parser.parse_args()
    print(args)

    process_folder(args.input, args.output)
    print("\n[INFO] All files processed successfully.")
