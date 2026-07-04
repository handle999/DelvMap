# 坐标系 bug 修复 — 纬度反转与尺度对齐

> 主题：didi_xian ↔ DelvMap DSFNet 全图坐标映射的两个 bug 及修复。
> 影响文件：`adaptive_map_completion/didi_bridge.py`

## 背景

AMC 评估链路需要把 DSFNet 全图预测（5625×6610，覆盖 Xi'an bbox）裁到 didi_xian 的 400×400 region，转成 inferred_rn 喂给 AMC。最初评估 AMC APLS 只有 0.276（partial baseline 0.682），怀疑坐标映射有问题。

## Bug 1：region_bbox 纬度反转（致命）

### 现象
inferred 与 GT 的像素 IoU 仅 0.003，8 个 region 全部灾难性低（IoU 0.005-0.056）。DSFNet 全图预测裁窗和 didi sat 空间相关性 ≈ 0。

### 根因
`didi_bridge.region_bbox` 的纬度方向和 samroad `download_use_osm.py` 的 **NW-first** 编号反了：

| | 代码 | 含义 |
|---|---|---|
| 修复前 | `lat_st = lat_min + i*dlat` | i=0 在最南（lat_min 端）|
| 修复后 | `lat_ed = lat_max - i*dlat` | i=0 在最北（lat_max 端）|

samroad 的 region 编号是 NW-first（tile 从左上角 NW 开始行优先 TL→BR，i=0=最北行）。`didi_bridge` 原实现把 i=0 当最南，导致每个 region 的 DSFNet 裁窗被映射到整图**纬度对称位置的另一块地**——region_92 偏移 0.034°（约半图）。

### 定位过程
1. 模板匹配 didi sat vs `rawdata/sat_img.png` 相关系数仅 0.17，疑似不同帧
2. 复现 samroad `get_local_sat` 采样逻辑，从 `rawdata/sat_img.png` 采 region_92 → 与实际 `region_92_sat.png` 像素差 0.1、相关 1.000，**证明 sat 确实来自 sat_img.png，采样逻辑对**
3. 那个 0.17 是因为比对时用了 `didi_bridge.region_bbox` 算的裁窗位置——纬度反了，裁到了错误位置
4. 对比 samroad NW-first vs `didi_bridge` 的 lat 计算，发现方向相反

### 修复
`region_bbox` 改为 NW-first（`adaptive_map_completion/didi_bridge.py`）：
```python
lat_ed = cfg['lat_max'] - i * dlat          # 北边 (i=0 = lat_max)
lat_st = cfg['lat_max'] - (i + 1) * dlat    # 南边
```

### 验证
修复后 region_92 inferred vs GT：IoU 0.003 → 0.252，Precision 0.998。全量 AMC APLS 0.276 → 0.562。

## Bug 2：裁窗尺度不一致（次要）

### 现象
DSFNet 全图原生 ~1.23 m/px（5625×6610 覆盖 Xi'an bbox），didi region 是 400×400 像素覆盖同一 geo bbox（1.0 m/px）。裁出的 region 子图是 325×325 px，但骨架化后用 `rc_to_latlon(size=400)` 映射，把 325px 内容当 400px 映射 → inferred 被压到 region 左上 81%。

### 修复
`dsfnet_region_inferred` 裁窗后 **resize 到 400×400 再骨架化**，让 inferred 像素原生落在 [0,400)，与 partial/GT 共享同一套 400px 栅格：
```python
if region_pred.shape[0] != region_size or region_pred.shape[1] != region_size:
    region_pred = cv2.resize(region_pred, (region_size, region_size),
                             interpolation=cv2.INTER_LINEAR)
```

## 注意事项

- `rc_to_latlon` / `latlon_to_rc` 是 region 内部 (row,col)↔(lat,lon) 的线性映射，row=0 恒为北边（lat_ed），**不依赖 region 在整图的位置**，所以 NW-first 修正后这两个函数无需改动。
- `rawdata/sat_img.png` 是 Web Mercator 投影，但在 Xi'an 0.07° 小范围内 Mercator vs 线性映射差异 <1px，可忽略（曾误判为根因，已排除）。
