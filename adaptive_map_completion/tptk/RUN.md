# 1. 字段

```txt
字段位置,样例值,对应代码变量,含义解析
0,#,-,轨迹开始标记。
1,1db9f...071340,traj.tid,轨迹ID (Trajectory ID)：通常由 OID + 起始时间 + 结束时间拼接而成，保证全局唯一。
2,1db9f3...008a62,traj.oid,对象ID (Object ID)：通常是加密后的用户ID或车辆ID。
3,2018/09/30 07:05:55,pt_list[0].time,开始时间。
4,2018/09/30 07:13:40,pt_list[-1].time,结束时间。
5,2.6022 km,traj.get_length(),轨迹总长度。

索引,样例值,对应代码变量,含义解析
0,2018/09/30 07:05:55,pt.time,时间戳：当前采样点的时间。
1,34.2799809,pt.lat,原始纬度：GPS观测的真实纬度。
2,108.9424189,pt.lng,原始经度：GPS观测的真实经度。
3,2979,candi_pt.eid,路段ID (Edge ID)：该点被算法匹配到的底层路网路段编号。
4,34.2799818,candi_pt.lat,投影纬度：将GPS点垂直投影到匹配路段上的纬度坐标。
5,108.9424751,candi_pt.lng,投影经度：将GPS点垂直投影到匹配路段上的经度坐标。
6,5.16,candi_pt.error,匹配误差：原始GPS点距离投影点（路网）的直线物理距离（米）。
7,201.84,candi_pt.offset,路段偏移量：投影点距离当前路段 (eid) 起点（上一个路口）的沿着道路走过的距离（米）。这个值在计算寻路距离时非常关键。
```

```shell
conda install gdal "numpy<2.0"

# 还会报错
Traceback (most recent call last):
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 30, in swig_import_helper
    return importlib.import_module(mname)
  File "/home//miniconda3/envs/SAM/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1050, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1027, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1006, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 674, in _load_unlocked
  File "<frozen importlib._bootstrap>", line 571, in module_from_spec
  File "<frozen importlib._bootstrap_external>", line 1176, in create_module
  File "<frozen importlib._bootstrap>", line 241, in _call_with_frames_removed
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found (required by /home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/../../../libgdal.so.37)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home//sam_road/utils/tptk/main.py", line 2, in <module>
    from common.road_network import load_rn_shp
  File "/home//sam_road/utils/tptk/common/road_network.py", line 3, in <module>
    from osgeo import ogr
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 35, in <module>
    _gdal = swig_import_helper()
  File "/home//miniconda3/envs/SAM/lib/python3.10/site-packages/osgeo/__init__.py", line 32, in swig_import_helper
    return importlib.import_module('_gdal')
  File "/home//miniconda3/envs/SAM/lib/python3.10/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ModuleNotFoundError: No module named '_gdal'
````

报错含义：你新安装的 `GDAL (libgdal.so.37)` 是用较新的 `GCC` 编译器编译的，它需要版本为 `CXXABI_1.3.15` 的 `C++` 标准库。

冲突点：报错显示系统正在加载 `/lib/x86_64-linux-gnu/libstdc++.so.6`，这是你 `Linux` 系统自带的老版本库，它不支持这个新版本标准。而 `Conda` 环境通常应该优先使用自己的新版库。

所以使用
```shell
conda install -c conda-forge libstdcxx-ng
```

还是不行，所以要尝试降低 `GDAL` 版本

```shell
conda remove gdal libgdal
conda install "gdal=3.6.2" "numpy>=1.21"
```

# xian

## compare
| param(min) | time-seg | time-stat |  objects | points | trajectories |
| :-----: | :----: | :----: | :----: | :---: | :---: |
| raw | - | 1:51:24 | 6,600,129 | 600,845,426 | 8,557,131 |
| 5 | 3:14:02 | 1:50:17 | 6,588,432 | 600,639,126 | 9,019,818 |
| 1 | 2:29:21 | 2:29:21 | 6,572,027 | 592,975,777 | 23,826,622 |

## command
```shell
cd ./utils
# (SAM) hanhaoyu@debian:~/sam_road/utils$

python -m tptk.main --phase segment_mm --mm_traj_dir ../xian/filtered_mm_traj --segment_output_dir ../xian/seg_mm_traj_5

python -m tptk.main --phase stat --clean_traj_dir ../xian/seg_mm_traj_5/

python -m tptk.common.hhy_txt_to_csv -i ../xian/seg_mm_traj -o ../xian/seg_mm_traj_csv

python -m tptk.common.hhy_mm_txt_to_csv -i ../xian/filtered_mm_traj -o ../xian/filtered_mm_traj_csv

python -m tptk.common.hhy_mm_txt_to_csv -i ../xian/seg_mm_traj_5 -o ../xian/seg_mm_traj_5_csv

```

## raw

```shell
Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/filtered_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:51:24<00:00,  4.68s/it]
#objects:6600129
#points:600845426
#trajectories:8557131

```

## seg-5min

```shell

Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../xian/filtered_mm_traj', segment_output_dir='../xian/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../xian/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:14:02<00:00,  8.15s/it]
Segmentation done.


Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:50:17<00:00,  4.63s/it]
#objects:6588432
#points:600639126
#trajectories:9019818


Namespace(input='../xian/seg_mm_traj', output='../xian/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [1:59:06<00:00,  5.00s/it]

[INFO] All files processed successfully.
```

## seg-1min

```shell
Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../xian/filtered_mm_traj', segment_output_dir='../xian/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../xian/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:09:35<00:00,  7.97s/it]
Segmentation done.

Namespace(tdrive_root_dir=None, clean_traj_dir='../xian/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [2:29:21<00:00,  6.28s/it]
#objects:6572027
#points:592975777
#trajectories:23826622

Namespace(input='../xian/seg_mm_traj', output='../xian/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [2:06:06<00:00,  5.30s/it]

[INFO] All files processed successfully.

```

# chengdu

```shell
cd ./utils

python -m tptk.main \
    --phase segment_mm \
    --mm_traj_dir ../chengdu/filtered_mm_traj \
    --segment_output_dir ../chengdu/seg_mm_traj

Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path=None, mm_traj_dir='../chengdu/filtered_mm_traj', segment_output_dir='../chengdu/seg_mm_traj', phase='segment_mm')
Start segmenting MM trajectories from ../chengdu/filtered_mm_traj...
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1451/1451 [4:38:02<00:00, 11.50s/it]
Segmentation done.

```

```shell
cd ./utils

python -m tptk.main --phase stat --clean_traj_dir ../chengdu/seg_mm_traj/

Namespace(tdrive_root_dir=None, clean_traj_dir='../chengdu/seg_mm_traj/', rn_path=None, mm_traj_dir=None, segment_output_dir=None, phase='stat')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1449/1449 [2:46:45<00:00,  6.90s/it]
#objects:11672503
#points:872526100
#trajectories:14452855

```

```shell
cd ./utils

python -m tptk.common.hhy_txt_to_csv -i ../chengdu/seg_mm_traj -o ../chengdu/seg_mm_traj_csv

Namespace(input='../chengdu/seg_mm_traj', output='../chengdu/seg_mm_traj_csv')
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1449/1449 [2:53:22<00:00,  7.18s/it]

[INFO] All files processed successfully.

```

# traj to path

Note: the enter time of the first path entity & the leave time of the last path entity is not accurate
原来的 `osm2rn/osm_to_rn.py` 存在问题，会覆盖双向道路，所以重新写了一个 `osm2rn/osm_to_rn_double.py` ，这样生成的没啥问题
等一下还是有问题，eid编号不一致了，所以得用原始 map matching 使用的 shpfile 去进行获取。这样才会完全没问题

```shell
# AttributeError: module 'networkx' has no attribute 'read_shp'
# 需要 networkx < 3
conda install networkx=2.8
```


```shell
cd ./utils

python -m tptk.main --phase construct_path --mm_traj_dir ../xian/seg_mm_traj_5 --path_output_dir ../xian/seg_mm_path_5 --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp

(SAM) E:\School\2025\20250311Road\GraphBased\sam_road\utils>python -m tptk.main --phase construct_path --mm_traj_dir ../xian/seg_mm_traj_5 --path_output_dir ../xian/seg_mm_path_5 --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp
Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path='../xian/osm/rn-comp-xa-190101-didi/edges.shp', mm_traj_dir='../xian/seg_mm_traj_5', segment_output_dir=None, path_output_dir='../xian/seg_mm_path_5', phase='construct_path')
Loading road network from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
C:\Users\Highee\.conda\envs\SAM\lib\site-packages\osgeo\ogr.py:593: FutureWarning: Neither ogr.UseExceptions() nor ogr.DontUseExceptions() has been explicitly called. In GDAL 4.0, exceptions will be enabled by default.
  warnings.warn(
# of nodes:4764
# of edges:10760
Start constructing paths from ../xian/seg_mm_traj_5...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:36:05<00:00,  9.08s/it]
Route construction done.

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.main --phase construct_path --mm_traj_dir ../xian/seg_mm_traj_5 --path_output_dir ../xian/seg_mm_path_5 --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp
Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path='../xian/osm/rn-comp-xa-190101-didi/edges.shp', mm_traj_dir='../xian/seg_mm_traj_5', segment_output_dir=None, path_output_dir='../xian/seg_mm_path_5', phase='construct_path')
Loading road network from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
# of nodes:4764
# of edges:10760
Start constructing paths from ../xian/seg_mm_traj_5...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:13:59<00:00,  8.15s/it]
Route construction done.
```

```shell
python -m tptk.main --phase construct_path --mm_traj_dir ../xian/filtered_mm_traj --path_output_dir ../xian/filtered_mm_path --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.main --phase construct_path --mm_traj_dir ../xian/filtered_mm_traj --path_output_dir ../xian/filtered_mm_path --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp
Namespace(tdrive_root_dir=None, clean_traj_dir=None, rn_path='../xian/osm/rn-comp-xa-190101-didi/edges.shp', mm_traj_dir='../xian/filtered_mm_traj', segment_output_dir=None, path_output_dir='../xian/filtered_mm_path', phase='construct_path')
Loading road network from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
# of nodes:4764
# of edges:10760
Start constructing paths from ../xian/filtered_mm_traj...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1428/1428 [3:14:26<00:00,  8.17s/it]
Route construction done.
```

# path to csv (LineString)

## seg_mm_5
```shell
cd ./utils

python -m tptk.common.hhy_path_txt_to_csv \
    -i ../xian/seg_mm_path_5 \
    -o ../xian/seg_mm_path_5_csv \
    --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.common.hhy_path_txt_to_csv \
    -i ../xian/seg_mm_path_5 \
    -o ../xian/seg_mm_path_5_csv \
    --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp
Namespace(input='../xian/seg_mm_path_5', output='../xian/seg_mm_path_5_csv', rn_path='../xian/osm/rn-comp-xa-190101-didi/edges.shp')
[INFO] Loading road network from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
[INFO] Loaded 10760 edges with geometries.
Converting Paths to CSV: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1427/1427 [1:28:02<00:00,  3.70s/it]

[INFO] All files processed successfully.
```

## filtered_mm
```shell
python -m tptk.common.hhy_path_txt_to_csv \
    -i ../xian/filtered_mm_path \
    -o ../xian/filtered_mm_path_csv \
    --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.common.hhy_path_txt_to_csv \
    -i ../xian/filtered_mm_path \
    -o ../xian/filtered_mm_path_csv \
    --rn_path ../xian/osm/rn-comp-xa-190101-didi/edges.shp
Namespace(input='../xian/filtered_mm_path', output='../xian/filtered_mm_path_csv', rn_path='../xian/osm/rn-comp-xa-190101-didi/edges.shp')
[INFO] Loading road network from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
[INFO] Loaded 10760 edges with geometries.
Converting Paths to CSV: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1427/1427 [1:27:41<00:00,  3.69s/it]

[INFO] All files processed successfully.
```

# path to shp

## seg_mm_5
```shell
python -m tptk.common.hhy_extract_active_shp \
    --path_dir ../xian/seg_mm_path_5 \
    --in_shp ../xian/osm/rn-comp-xa-190101-didi/edges.shp \
    --out_shp ../xian/osm/rn-comp-xa-190101-seg5/edges.shp

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.common.hhy_extract_active_shp --path_dir ../xian/seg_mm_path_5 --in_shp ../xian/osm/rn-comp-xa-190101-didi/edges.shp --out_shp ../xian/osm/rn-comp-xa-190101-active/edges.shp
[INFO] Scanning ../xian/seg_mm_path_5 for active edge IDs...
Parsing Paths: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1427/1427 [02:21<00:00, 10.08it/s]
[INFO] Found 8839 unique active eids.
[INFO] Created output directory: ../xian/osm/rn-comp-xa-190101-active
[INFO] Extracting subset from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
Filtering SHP: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10760/10760 [00:03<00:00, 3449.42it/s]
[INFO] Successfully saved 8839 features to ../xian/osm/rn-comp-xa-190101-seg5/edges.shp.
```

## filtered_mm
```shell
python -m tptk.common.hhy_extract_active_shp \
    --path_dir ../xian/filtered_mm_path \
    --in_shp ../xian/osm/rn-comp-xa-190101-didi/edges.shp \
    --out_shp ../xian/osm/rn-comp-xa-190101-filter/edges.shp

(SAM) hanhaoyu@debian:~/sam_road/utils$ python -m tptk.common.hhy_extract_active_shp     --path_dir ../xian/filtered_mm_path     --in_shp ../xian/osm/rn-comp-xa-190101-didi/edges.shp     --out_shp ../xian/osm/rn-comp-xa-190101-filter/edges.shp
[INFO] Scanning ../xian/filtered_mm_path for active edge IDs...
Parsing Paths: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1427/1427 [02:27<00:00,  9.67it/s]
[INFO] Found 8880 unique active eids.
[INFO] Created output directory: ../xian/osm/rn-comp-xa-190101-filter
[INFO] Extracting subset from ../xian/osm/rn-comp-xa-190101-didi/edges.shp...
Filtering SHP: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10760/10760 [00:03<00:00, 3564.19it/s]
[INFO] Successfully saved 8880 features to ../xian/osm/rn-comp-xa-190101-filter/edges.shp.
```
