# DelvMap 数据制备完整指南

本文档详细介绍如何为 DelvMap 两阶段模型准备训练数据。

---

## 目录

1. [数据流程概览](#1-数据流程概览)
2. [原始数据要求](#2-原始数据要求)
3. [第一步：生成基础数据 (prepare_dataset.py)](#3-第一步生成基础数据-prepare_datasethttpprepare_datasethttp))
4. [第二步：裁剪为训练 patch (create_dataset.py)](#4-第二步裁剪为训练-patch-create_datasethttpcreate_datasethttp))
5. [运行示例](#5-运行示例)
6. [输出目录结构](#6-输出目录结构)
7. [常见问题](#7-常见问题)

---

## 1. 数据流程概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              原始数据准备                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. 卫星影像 (RGB, PNG)                    → sat_img.png                   │
│  2. 轨迹热力图 (二值, PNG)                  → traj_heat.png                 │
│  3. OSM PBF 数据 (道路+建筑)                → xian-plus-190101.osm.pbf      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              步骤1: prepare_dataset.py                                      │
│              生成完整大图 (6625 x 6610)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输出目录: dataset/delvmap_data/                                           │
│  ├── basemap/0.png          - 现有道路 (highway, 二值化)                    │
│  ├── src/0.png              - 卫星影像                                      │
│  ├── traj/0.png             - 轨迹路径                                      │
│  ├── trajpoint/0.png        - 轨迹点                                        │
│  ├── building_label/0.png  - 建筑 (building, 二值化)                       │
│  └── map_label/                                                    │
│      └── label_width2/0.png - 真值道路                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              步骤2: create_dataset.py                                       │
│              滑动窗口裁剪为 256x256 patch                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输出目录: dataset/delvmap_data_patches/                                   │
│  ├── src/                    - 6通道numpy数组                               │
│  ├── label/                  - 道路标签 PNG                                 │
│  ├── building_label/         - 建筑标签 PNG                                 │
│  ├── visualize/              - 可视化                                        │
│  └── ...                                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 原始数据要求

### 2.1 数据来源

| 数据类型 | 来源 | 格式 |
|----------|------|------|
| 卫星影像 | 天地图/Google Earth/高德 API | PNG, RGB, 3通道 |
| 轨迹热力图 | 快递员 GPS 轨迹可视化 | PNG, 单通道, 二值化 (0/255) |
| OSM 数据 | [Geofabrik](https://download.geofabrik.de/) | .osm.pbf |

### 2.2 你的数据存放位置

```
DelvMap/
├── rawdata/
│   ├── sat_img.png          # 卫星影像 (6625 x 6610)
│   └── traj_heat.png        # 轨迹热力图 (二值化)
│
└── dataset/
    └── osm/
        └── xian-plus-190101.osm.pbf   # OSM 数据 (已有)
```

### 2.3 地理范围配置 (来自 RUN.md)

```
图像尺寸: 6625 x 6610 (宽 x 高)

路网有效范围 (WGS):
- 纬度: 34.206385 ~ 34.279658
- 经度: 108.917423 ~ 108.99286
```

---

## 3. 第一步：生成基础数据 (prepare_dataset.py)

### 3.1 脚本功能

从原始数据生成 `create_dataset.py` 需要的完整大图：

- 从 OSM PBF 提取 **道路** (`highway=*`)
- 从 OSM PBF 提取 **建筑** (`building=*`)
- 处理轨迹数据
- 所有数据二值化 (0/255)

### 3.2 配置修改

如需修改地理范围或图像尺寸，编辑 `prepare_dataset.py` 中的 `CONFIG`:

```python
CONFIG = {
    # 地理范围
    'lat_min': 34.206385,
    'lat_max': 34.279658,
    'lon_min': 108.917423,
    'lon_max': 108.99286,

    # 图像尺寸
    'img_w': 5625,
    'img_h': 6610,

    # OSM PBF 文件
    'osm_pbf': r"dataset\osm\xian-plus-190101.osm.pbf",

    # 道路宽度 (像素)
    'road_width': 2,

    # 路径
    'rawdata_dir': r"rawdata",
    'output_dir': r"dataset\delvmap_data",
}
```

### 3.3 运行命令

```bash
cd e:\School\2025\20250311Road\GraphBased\DelvMap
python dataset/prepare_dataset.py
```

### 3.4 输出

```
dataset/delvmap_data/
├── basemap/0.png           # 现有道路 (从 OSM highway 提取)
├── src/0.png               # 卫星影像
├── traj/0.png              # 轨迹路径 (闭运算连接)
├── trajpoint/0.png         # 轨迹点 (你的 traj_heat)
├── building_label/0.png    # 建筑 (从 OSM building 提取)
└── map_label/
    └── label_width2/
        └── 0.png           # 真值道路
```

---

## 4. 第二步：裁剪为训练 patch (create_dataset.py)

### 4.1 脚本功能

- 从完整大图裁剪出 256x256 的训练 patch
- 支持两种模式：**随机裁剪** 或 **滑动窗口**
- 滑动窗口模式：边缘自动补零填充
- 自动划分训练集/验证集

### 4.2 命令行参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | (必填) | 输入数据根目录 |
| `--output` | `-o` | input_patches | 输出目录 |
| `--mode` | `-m` | random | 裁剪模式: `random` 或 `sliding` |
| `--stride` | - | 256 | 滑动窗口步长 |
| `--patch_size` | `-s` | 256 | patch 大小 |
| `--patch_num` | `-n` | 100 | 每张图采样数 (仅 random 模式) |
| `--split_ratio` | `-r` | 7/8 | 训练集比例 (仅 random 模式) |
| `--subset` | `-u` | all | 生成哪个集合: `train`/`val`/`all` |
| `--train_ratio` | - | 0.8 | 训练集比例 (sliding 模式有效) |
| `--val_ratio` | - | 0.1 | 验证集比例 (sliding 模式有效) |

### 4.3 运行示例

#### 随机裁剪模式 (默认)

```bash
# 生成训练集和验证集
python dataset/create_dataset.py -i dataset/delvmap_data -n 100

# 指定输出目录
python dataset/create_dataset.py -i dataset/delvmap_data -o my_patches -n 50

# 只生成验证集
python dataset/create_dataset.py -i dataset/delvmap_data -u val -n 50
```

#### 滑动窗口模式 (推荐)

```bash
# 无重叠滑动窗口
python dataset/create_dataset.py -i dataset/delvmap_data -m sliding

# 有重叠滑动窗口 (步长128 = 50%重叠)
python dataset/create_dataset.py -i dataset/delvmap_data -m sliding --stride 128

# 指定输出目录
python dataset/create_dataset.py -i dataset/delvmap_data -o my_patches -m sliding --stride 128
```

---

## 5. 运行示例

### 完整流程

```bash
# 步骤1: 生成基础数据
cd e:\School\2025\20250311Road\GraphBased\DelvMap
python dataset/prepare_dataset.py

# 步骤2: 裁剪为训练 patch (滑动窗口，无重叠)
python dataset/create_dataset.py -i dataset/delvmap_data -m sliding

# 步骤2: 裁剪为训练 patch (滑动窗口，50%重叠)
python dataset/create_dataset.py -i dataset/delvmap_data -m sliding --stride 128
```

---

## 6. 输出目录结构

### 6.1 prepare_dataset.py 输出

```
dataset/delvmap_data/
├── basemap/              # 现有道路 (完整大图)
│   └── 0.png
├── src/                  # 卫星影像 (完整大图)
│   └── 0.png
├── traj/                 # 轨迹路径 (完整大图)
│   └── 0.png
├── trajpoint/            # 轨迹点 (完整大图)
│   └── 0.png
├── building_label/       # 建筑掩膜 (完整大图)
│   └── 0.png
└── map_label/
    └── label_width2/     # 真值道路 (完整大图)
        └── 0.png
```

### 6.2 create_dataset.py 输出 (patch 格式)

```
dataset/delvmap_data_patches/
├── src/                      # 6通道numpy数组 [N, 256, 256, 6]
│   ├── 0.npy
│   ├── 1.npy
│   └── ...
│
├── split_indices.json        # train/val/test 划分索引 (sliding 模式生成)
│
├── label/                    # 道路标签 PNG (单通道)
│   ├── 0.png
│   ├── 1.png
│   └── ...
│
├── building_label/           # 建筑标签 PNG (单通道)
│   ├── 0.png
│   └── ...
│
├── visualize/                # 可视化 (label * 50)
│   ├── 0.png
│   └── ...
│
├── basemap_split/            # 基础路网
├── traj_split/               # 轨迹特征
├── trajpoint_split/          # 轨迹点
├── traj_and_point_split/     # 轨迹+点 (2通道)
└── traj_and_point_and_img_split/  # 轨迹+影像 (5通道)
```

### 6.3 数据格式说明

| 文件 | 格式 | 通道数 | 值范围 | 说明 |
|------|------|--------|--------|------|
| `src/*.npy` | numpy | 6 | 0-255 | [basemap, traj, trajpoint, src_BGR] |
| `label/*.png` | PNG | 1 | 0/255 | 真值道路 |
| `building_label/*.png` | PNG | 1 | 0/255 | 建筑掩膜 |

---

## 7. 常见问题

### Q1: 运行 prepare_dataset.py 报错 "No module named 'osmium'"

**解决**:
```bash
pip install osmium
```

### Q2: 轨迹热力图和卫星影像尺寸不一致

**解决**: `prepare_dataset.py` 会自动调整轨迹图尺寸以匹配卫星影像。

### Q3: 建筑掩膜是空的

**解决**: 检查 OSM PBF 文件是否包含 building 数据，或运行 `generate_building_mask.py` 单独生成。

### Q4: 滑动窗口模式的边界处理

**解决**: 边缘区域会自动用 **零填充** 到 256x256，确保覆盖全图。

### Q5: 如何模拟缺失路网

**解决**: 目前 `basemap` 和 `label` 使用相同的完整路网。如果需要模拟缺失路网，可以手动删除 `basemap/0.png` 中的部分道路（用图像编辑软件或代码），然后重新运行 `create_dataset.py`。

---

## 附录：依赖安装

```bash
pip install numpy opencv-python tqdm osmium
```

或者使用 conda:

```bash
conda install -c conda-forge osmium
```

---

## 附录二：数据格式与 DataLoader 处理流程

### 1. 数据文件格式

| 数据类型 | 文件格式 | Shape | Dtype | 值范围 | Unique |
|----------|----------|-------|-------|--------|--------|
| traj | .npy | (256,256,2) | uint8 | [0,255] | [0,255] |
| src | .png RGB | (256,256,3) | uint8 | [0,255] | [0,255] |
| label | .png gray | (256,256) | uint8 | [0,255] | [0,255] |
| building | .png gray | (256,256) | uint8 | [0,255] | [0,255] |

### 2. DataLoader 处理流程 (data_loader.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              输入: 原始文件                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  traj/*.npy      → np.load() → uint8 [0,255]                               │
│  src/*.png       → cv2.imread() → uint8 [0,255]                            │
│  label/*.png     → cv2.imread(gray) → uint8 [0,255]                        │
│  building/*.png  → cv2.imread(gray) → uint8 [0,255]                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│                              处理步骤                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【traj 处理】                                                              │
│  Step1: np.asarray(np.load(path)) + dtype=float                           │
│         → float64 [0, 255]                                                 │
│  Step2: transforms.ToTensor() + .float()                                   │
│         → torch.Tensor [2, H, W], float32 [0, 1]                           │
│                                                                             │
│  【src 处理】                                                               │
│  Step1: cv2.imread() + np.array(dtype=float)                              │
│         → float64 [0, 255]                                                 │
│  Step2: transforms.ToTensor() + .float()                                   │
│         → torch.Tensor [3, H, W], float32 [0, 1]                           │
│                                                                             │
│  【label 处理】                                                             │
│  Step1: cv2.imread(gray) → uint8 [0, 255]                                  │
│  Step2: np.expand_dims(label, axis=-1) → (H,W,1)                          │
│  Step3: transforms.ToTensor() + .float()  【正确】                         │
│         → torch.Tensor [1, H, W], float32 [0, 1]                           │
│         unique: [0.0, 1.0]                                                 │
│                                                                             │
│  【错误示例 - 不要这样做】                                                  │
│  Step3: transforms.ToTensor() + .float() / 255.0                          │
│         → torch.Tensor [1, H, W], float32 [0, 0.004]                       │
│         unique: [0.0, 0.0039]  ← 错误！会导致 metrics 全 0                  │
│                                                                             │
│  【building 处理】 (同 label)                                              │
│  Step1: cv2.imread(gray) → uint8 [0, 255]                                  │
│  Step2: np.expand_dims()                                                   │
│  Step3: transforms.ToTensor() + .float()  【正确】                         │
│         → torch.Tensor [1, H, W], float32 [0, 1]                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. 关键点：ToTensor() 的自动转换

`transforms.ToTensor()` 会自动执行：
1. 将 HWC 转换为 CHW (对于图像)
2. 将 [0,255] uint8 转换为 [0,1] float

**重要：不需要额外除以 255！**

### 4. 修复记录

| 日期 | 问题 | 修复 |
|------|------|------|
| 2026-05-13 | label 经过 `/255` 后值变成 0 或 0.0039，导致 metrics 全 0 | 移除 `/255.0`，直接使用 `ToTensor()` |
| 2026-05-14 | translator.py中temp张量维度错误导致RuntimeError | 修复 `torch.ones_like(sb_out1[:, :1, :, :])` |

### 5. 数据验证脚本

如需验证数据集是否正确加载，可使用：

```bash
# 分析单个样本的数据流
python analyze_data.py

# 完整流水线测试
python test_pipeline.py
```

这些脚本会打印：
- 原始文件的shape、dtype、值范围
- DataLoader处理后的值
- 模型输出和Metrics计算结果