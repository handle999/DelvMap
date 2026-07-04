import os
import csv


def merge_csv_files(input_folder, output_file):
	# 获取文件夹下所有 csv 文件
	csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
	csv_files.sort()  # 排序确保合并顺序

	if not csv_files:
		print("文件夹内没有找到 CSV 文件！")
		return

	first_file = True
	total_count = 0

	with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
		# 定义表头字段（与你提供的示例一致）
		fieldnames = ['tid', 'oid', 'start_time', 'end_time', 'wkt']
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)

		# 写入总表头
		writer.writeheader()

		for filename in csv_files:
			file_path = os.path.join(input_folder, filename)
			print(f"正在读取: {filename}")

			with open(file_path, 'r', encoding='utf-8') as infile:
				# 使用 DictReader 自动处理列对齐
				reader = csv.DictReader(infile)

				row_count = 0
				for row in reader:
					# 写入一行数据
					writer.writerow(row)
					row_count += 1

				total_count += row_count

	print(f"\n合并完成！")
	print(f"生成的总文件: {output_file}")
	print(f"总计合并轨迹数: {total_count}")


if __name__ == "__main__":
	# --- 请根据实际路径修改 ---
	INPUT_DIR = r'E:\School\2025\20251022RSTraj\dataset\clean\vis\didi_gaia\xian'
	OUTPUT_FILE = r'E:\School\2025\20251022RSTraj\dataset\clean\vis\didi_gaia\xian_all_merged.csv'

	merge_csv_files(INPUT_DIR, OUTPUT_FILE)
