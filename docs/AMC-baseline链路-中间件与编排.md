# AMC baseline 链路 — 中间件与编排

> 主题：didi_xian 评估链路的中间件 (`didi_bridge.py`) 与编排 (`run_didi_eval.py`) 设计。
> AMC（Adaptive Map Completion）作为 baseline，不优化其内部逻辑。

## 链路总览

```
partial pickle ─→ WGS84 partial_rn ─┐
                                     │
DSFNet 整图预测 → 裁 region → resize400 → 骨架化 → inferred_rn ─┤
                                     │   ↓
                                     ├─→ AMC.adaptive_fuse (P2 空间合并, trajs=[])
                                     │   ↓ fused_rn
                                     │   ↓
                fused_rn → 邻接字典 pickle → metrics/eval.py (APLS + TOPO)
```

## didi_bridge.py（中间件）

职责：didi_xian pickle ↔ DelvMap AMC 的格式/坐标转换。

### 核心函数
| 函数 | 作用 |
|------|------|
| `region_bbox(idx)` | region 编号 → geo bbox（NW-first，见 [坐标系bug修复](坐标系bug修复-纬度反转与尺度对齐.md)）|
| `rc_to_latlon` / `latlon_to_rc` | region 内 (row,col) ↔ (lat,lon) 线性映射（row=0 北，col=0 西）|
| `bbox_to_whole_pixel` | region geo bbox → DelvMap 整图像素窗 |
| `pickle_to_rn` | didi 邻接字典 {(row,col):[(row,col)...]} → UndirRoadNetwork (WGS84) |
| `rn_to_pickle` | UndirRoadNetwork → didi 邻接字典（量化 0.001px 去抖）|
| `dsfnet_region_inferred` | DSFNet 全图预测 → region inferred_rn（resize 到 400×400 再骨架化）|
| `load_didi_pickle` | 加载 partial/gt/active pickle |

### 坐标系约定
- region (row,col) ↔ WGS84：`lat = lat_ed - (row/size)*(lat_ed-lat_st)`，`lon = lon_st + (col/size)*(lon_ed-lon_st)`
- row=0 在北（lat_ed），col=0 在西（lon_st），size=400
- 与 samroad `download_use_osm.py` 完全一致

## run_didi_eval.py（编排）

职责：串联单 region 链路 + 调 metrics 评估。

### 三种 pred_mode
| mode | 含义 | 用途 |
|------|------|------|
| `amc` | partial + DSFNet inferred → AMC fuse（主流程）| 主评估 |
| `partial` | partial pickle 直接当 pred | baseline 下限 |
| `gt` | GT pickle 直接当 pred | sanity（应 APLS=1.0）|

### run_one_region 流程（amc 模式）
1. partial pickle → partial_rn（AMC existing_rn / base）
2. partial 空（采样删光）→ SKIP（无 base 可补全）
3. trajs=[]（didi_xian 无 mm 轨迹，AMC 走纯 P2 空间就近合并）
4. DSFNet → inferred_rn（补全候选）
5. `DelvMapConnector.adaptive_fuse(partial_rn, inferred_rn, [])` → fused_rn
6. fused_rn → pickle（`graph/{idx}.p`）
7. 保存产物：`mask/{idx}_road.png`、`viz/{idx}.png`（对齐 samroad infer 结构，见 [可视化规范](可视化规范-samroad-triage配色与目录结构.md)）

### out_compressed=False（评估关键）
评估时 `DelvMapConnector(out_compressed=False)`。compress 会无差别合并所有 degree=2 中间节点（包括 partial 原本合理的节点），使 fused 节点稀疏、APLS 控制点对不齐。详见 [评估指标与诊断](评估指标与诊断-APLS-TOPO与碎片根因.md) §compress 节点密度问题。

## 用法

```bash
# 单 region 验证
python adaptive_map_completion/run_didi_eval.py --regions 92 --work_dir experiments/r92 --pred_mode amc

# 全量 test split
python adaptive_map_completion/run_didi_eval.py --split test --work_dir experiments/didi_amc_test --pred_mode amc

# GT 自评 (sanity, 应 APLS=1.0)
python adaptive_map_completion/run_didi_eval.py --regions 92 --work_dir experiments/sanity --pred_mode gt
```

## 依赖
- `metrics/eval.py`（本地，APLS + TOPO，见 [评估自含解耦](评估自含解耦-metrics本地化.md)）
- `datasets/didi/xian/`（本地，GT + partial + sat）
- DSFNet 推理产物 `results/{name}/all_{epoch}/images_full/`
