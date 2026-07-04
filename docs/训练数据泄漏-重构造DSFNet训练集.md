# 训练数据泄漏 — 重构造 DSFNet 训练集

> 主题：DSFNet 训练数据泄漏根因、基于 didi_xian 标准数据集重新构造训练集的方案。
> 影响文件：`dataset/prepare_didi_trainset.py`（新建）、`dataset/didi_xian_train/`（生成，不入 git）

## 问题发现

AMC baseline 修复 4-邻接碎片后（见 [评估指标与诊断](评估指标与诊断-APLS-TOPO与碎片根因.md)），TOPO recall 达 0.738，**超过 samroad rn-only 主设定（0.6909）**——一个 baseline 后处理不该比 samroad 精心训练的 TopoNet 还高，不合理。

排查发现 DSFNet 在 didi_xian test split 上 recall 异常高（0.784），多 region rec 0.95-1.0。这不是模型能力强，是**训练数据泄漏**。

## 泄漏根因

DelvMap 现有训练集 `dataset/xian_2019_delvmap/`（2288 个 256px patch，从 `rawdata/sat_img.png` **像素直切**）和 didi_xian test region（400px，**Mercator 重投影**采样）是两套不同的切分网格：

- DelvMap 256px 切分：5625×6610 大图，patch 256 stride 128，像素坐标直切
- didi_xian 400px region：21×18 网格，samroad `get_local_sat` Mercator 重投影采样

两者网格原点、间距、投影都不同。**DelvMap 的 train patch 完全可能落在 didi_xian 的 test region 内**——DSFNet 训练时见过这些卫星图。

### 泄漏量化
```
DelvMap 1601 个 train patch 中, 175 个 (10.9%) 落在 didi_xian test region 内
```
（用 `didi_bridge.region_bbox` + `raster_to_shp.geo_to_pixel` 反查 patch 中心 geo 坐标，与 didi_xian test region geo bbox 求交确认）

## 重构造方案

基于标准数据集 `datasets/didi/xian/` 重新构造训练集，消除泄漏。

### 核心思路
- **坐标系**：用 samroad `get_local_sat` 同款 Mercator 重投影，从 rawdata 采到 didi_xian region 400px 坐标系（与评估数据同源同公式）
- **256 切分**：每个 400px region sliding 2×2 切 4 个 256px patch（stride 144，overlap 112px，全覆盖）
- **split**：按 didi_xian `data_split.json` 的 **region 级**划分（train region 的 4 patch → train），零泄漏
- **patch idx**：`region_idx * 4 + offset_idx`（0=TL, 1=TR, 2=BL, 3=BR），可追溯

### 4 模态数据来源（全部对齐到 didi_xian region 400px Mercator 坐标系）
| 模态 | 来源 | 说明 |
|------|------|------|
| sat | `rawdata/sat_img.png` Mercator 重投影 | 与 didi_xian region_sat 同源同公式，像素差 0.06 |
| building_label | `rawdata/building_label_full.png` Mercator 重投影 | didi_xian 无 building，从 DelvMap 256patch 拼回全图再采样 |
| traj | `datasets/didi/xian/2019_400/region_{idx}_traj.png` | samroad `generate_traj.py` 已生成（同款 Mercator 采样）|
| label | `datasets/didi/xian/2019_400/region_{idx}_gt.png` | samroad GT，与评估 GT 同源，比 DelvMap label_full 更全 |

### 关键决策
1. **label 用 didi_xian region_gt**（非 DelvMap label_full）：region_gt 是 samroad 完整 GT，label_full 是其子集（recall 0.737）。用 region_gt 保证训练/评估 label 一致
2. **building_label 从 DelvMap 拼回全图采样**：didi_xian 无 building 数据，DSFNet 多任务损失需要 building 分支。`rawdata/building_label_full.png` 由 2288 个 256px patch（stride 128 max-overlap）拼回 5625×6610，与 sat 同 Mercator 坐标系
3. **traj npy point 通道置零**：DSFNet 原始 traj npy 是 (256,256,2)，ch0=traj 线，ch1=traj_point 点。didi_xian 无 point 数据，ch1 置零，模型自适应学习忽略

## 数据生成脚本：`dataset/prepare_didi_trainset.py`

### 流程
1. 读 rawdata 大图：`sat_img.png`（RGB）、`building_label_full.png`（灰度）
2. 对每个 didi_xian region（0-377，`didi_bridge.region_bbox` NW-first 算 geo bbox）：
   - sat/building 400px：Mercator 重投影采样（复用 samroad `get_local_sat` 公式）
   - traj/label 400px：直接读 didi_xian region 文件
3. 400→256 sliding 2×2 切分（offsets=[0,144]×[0,144]，4 patch/region，4 模态同 offset 同切）
4. 按 region 级 split 划分 patch，输出到 `dataset/didi_xian_train/`
5. 内置验证：泄漏检查、对齐检查、完整性检查

### 输出目录结构（对齐 DSFNet `data_loader.py:75-78`）
```
dataset/didi_xian_train/
├── traj_and_point_split/{idx}.npy   (256,256,2) uint8, ch0=traj, ch1=0
├── src_split/{idx}.png              256×256 RGB
├── label/{idx}.png                  256×256 灰度 (道路GT)
├── building_label/{idx}.png         256×256 灰度 (建筑GT)
└── split_indices.json               {train/val/test: [patch idx]}
```

### 验证结果（本地 dryrun）
- **零泄漏**：train 302 region，0 个落在 test region 内 ✓
- **sat 对齐**：生成 sat patch vs didi region_sat 像素差 0.06（<3）✓
- **完整性**：4 模态 shape 正确，traj ch0 有内容 ch1 置零，label/building 二值化正常 ✓
- **规模**：378 region → 1512 patch（train 1208 / val 148 / test 156）

## 训练（远程服务器）

```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py \
    --name didi_xian_train --dataroot ./dataset/didi_xian_train/ \
    --lam 0.2 --batch_size 8 --train_pattern DSFNet --net_trans DSFNet --model DSFNet \
    --gpu_ids 0 --n_epochs 200 --save_epoch_freq 5 --no_html
```

DSFNet 是 UNet 结构（`models/DSFNet_model.py`，`input_nc=5`：traj 2ch + src 3ch），`F.interpolate(scale_factor=2)` 上采样，不硬编码尺寸，支持 256px 输入。data_loader 读固定文件，文件多大训多大。

## 推理 + 评估

```bash
# 推理全图 (输出 results/didi_xian_train/all_{epoch}/images_full/)
python Dual_Signal_Fusion_based_Map_Completion/infer_all.py \
    --name didi_xian_train --dataroot ./dataset/didi_xian_train/ \
    --epoch latest --model DSFNet --net_trans DSFNet --train_pattern DSFNet --gpu_ids 0

# 重跑 AMC 评估 (链路自含, 见 评估自含解耦文档)
python adaptive_map_completion/run_didi_eval.py \
    --split test --work_dir experiments/didi_amc_test \
    --pred_dir results/didi_xian_train/all_latest/images_full --pred_mode amc
```

## 验证泄漏消除

重跑 AMC 评估后对比指标：

| 指标 | 泄漏时（旧） | 预期（泄漏消除后） |
|------|------------|------------------|
| TOPO recall | 0.738 | < 0.6909（samroad rn-only 基线）|
| APLS | 0.812 | 回落（合理）|
| TOPO F1 | 0.841 | 回落（合理）|

**关键判据**：TOPO recall 回落到 samroad rn-only（0.6909）以下，证明泄漏消除、DSFNet 表现真实可信。

## 相关文件
- `dataset/prepare_didi_trainset.py` — 数据生成脚本（含验证）
- `adaptive_map_completion/didi_bridge.py` — `region_bbox`（NW-first，见 [坐标系bug修复](坐标系bug修复-纬度反转与尺度对齐.md)）
- samroad `get_local_sat` 公式（`tools/prepare_dataset/download_use_osm.py:143-163`，照搬不跨项目依赖）
- `rawdata/building_label_full.png` — 256patch 拼回的建筑全图（生成训练集时用）
- DSFNet：`Dual_Signal_Fusion_based_Map_Completion/train.py`、`data_loader.py`、`models/DSFNet_model.py`

## 注意事项
- `dataset/didi_xian_train/` 是生成产物，**不入 git**（.gitignore 忽略），由 `prepare_didi_trainset.py` 重新生成
- `rawdata/building_label_full.png` 由 `xian_2019_delvmap/building_label/` 256patch 拼回，生成训练集前需先有此文件
- 若远程重新生成训练集，需先确保 `rawdata/sat_img.png` + `rawdata/building_label_full.png` + `datasets/didi/xian/` 齐全
