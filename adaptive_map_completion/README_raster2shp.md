# DSFNet → AdaMap 桥接：栅格 → Shapefile

把 DSFNet 输出的 256×256 patch 预测掩膜转换成 AdaMap 可直接消费的 WGS84 矢量路网 shapefile。

## 数据流概览

```
DSFNet ckpt (checkpoints/delvmap_exp2/)
    │
    ▼ infer_all.py            (全量 2288 patch 推理)
results/{name}/all_{epoch}/images_full/{idx}_pred_traj_img.png
    │
    ▼ raster_to_shp.py         (拼图 + 骨架化 + 矢量化 + WGS84)
inferred_rn_xa/edges.shp
    │
    ▼ run_adamap.py            (AdaMap 融合 OSM + 推断路网)
fused_rn_xa/edges.shp
```

## 环境

推荐用 `SAM` conda env，依赖完整（gdal、networkx 2.8、rtree、scikit-image、torch、cv2 都齐）。
但 scipy / skimage 加载需要更新版 `libstdc++`，所以**所有命令都加 `LD_PRELOAD`**：

```bash
export LD_PRELOAD=/home/hanhaoyu/miniconda3/envs/SAM/lib/libstdc++.so.6
export PY=/home/hanhaoyu/miniconda3/envs/SAM/bin/python
```

或者把 `LD_PRELOAD` 永久加到 `~/.bashrc` 里。

## 步骤 1：全量推理

```bash
$PY Dual_Signal_Fusion_based_Map_Completion/infer_all.py \
    --name delvmap_exp2 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --epoch 357 \
    --model DSFNet --net_trans DSFNet --train_pattern DSFNet \
    --gpu_ids 0
```

输出：
- `results/delvmap_exp2/all_357/images_full/{idx}_pred_traj_img.png` × 2288
- `results/delvmap_exp2/all_357/images_full/{idx}_pred_src_traj_img.png` × 2288

⚠ 注意：因为 train/val/test 是按 patch 随机切的（[split_indices.json](dataset/xian_2019_delvmap/split_indices.json) 是 7:1:2 散布），全量推理 70% 的 patch 在训练集里。这只是**为了拼图产物，不能用作模型评测**。

## 步骤 2：栅格 → Shapefile

```bash
$PY adaptive_map_completion/raster_to_shp.py \
    --pred_dir results/delvmap_exp2/all_357/images_full \
    --out_dir  inferred_rn_xa \
    --head pred_traj_img \
    --bin_thresh 128 \
    --min_obj 50 \
    --simplify_eps 2.0 \
    --save_full_pred
```

参数说明：
- `--head`：用哪个预测头当掩膜
  - `pred_traj_img`（默认）：traj+sat 融合，F1≈0.83
  - `pred_src_traj_img`：仅 sat，F1≈0.88，但和后续 AdaMap 的 traj 信号相关性更低
- `--bin_thresh`：二值化阈值（DSFNet 输出已经是 0/255，所以 128 即可）
- `--min_obj`：去除 < 50 像素的小连通块（噪点）
- `--simplify_eps`：Douglas-Peucker 简化阈值（像素），减小 shp 体积；2 px ≈ 3m
- `--save_full_pred`：把拼接后的全图灰度、二值、骨架都保存到 `out_dir`，便于人工检查

输出：
- `inferred_rn_xa/edges.shp`（+ `.dbf`/`.shx`），AdaMap 可直接 `load_rn_shp` 读
- `inferred_rn_xa/nodes.shp`
- `inferred_rn_xa/full_pred_pred_traj_img.png`（如指定 `--save_full_pred`）
- `inferred_rn_xa/binary.png`、`inferred_rn_xa/skeleton.png`

## 步骤 3：AdaMap 融合

### 3.1 不带轨迹（先打通管线，纯 P2 空间合并）

```bash
$PY adaptive_map_completion/run_adamap.py \
    --existing_rn dataset/osm/rn-comp-xa-190101-didi \
    --inferred_rn inferred_rn_xa \
    --out_dir     fused_rn_xa
```

### 3.2 带 mm 轨迹（启用 P1 轨迹支撑接边）

mm 轨迹数据需要先准备好，目录结构：

```
<mm_traj_root>/
├── <courier_id_1>/
│   ├── 2018-09-30.txt   (mm 格式，含 candi_pt.eid/error/offset)
│   └── ...
├── <courier_id_2>/
│   └── ...
```

```bash
$PY adaptive_map_completion/run_adamap.py \
    --existing_rn dataset/osm/rn-comp-xa-190101-didi \
    --inferred_rn inferred_rn_xa \
    --mm_traj_dir <mm_traj_root> \
    --min_trans_cnt 1 \
    --out_dir     fused_rn_xa
```

mm 轨迹原始数据在 `~/sam_road/datasets/didi/xian/filtered_mm_traj.zip`（独立工作项，不在本桥接脚本范围内）。

## 验证步骤

| # | 验证目标 | 操作 |
|---|---|---|
| 1 | 像素↔WGS 自洽 | `raster_to_shp.py` 启动时 `_self_test_coord()` 自动跑（往返误差 < 1e-6 像素） |
| 2 | 拼图正确 | 加 `--save_full_pred`，把 `full_pred_*.png` 与 [rawdata/sat_img.png](rawdata/sat_img.png) 在 QGIS / GIMP 叠加看路网形状 |
| 3 | 矢量化合理 | 看 `[rn] # nodes / # edges` 打印，应 ~10⁴ 量级，连通块数 << 像素噪点量级 |
| 4 | shp 兼容性 | `python -c "from tptk.common.road_network import load_rn_shp; rn = load_rn_shp('inferred_rn_xa/'); print(rn.number_of_edges())"` 能反读且边数一致 |
| 5 | GIS 叠加 | QGIS 加载 `inferred_rn_xa/edges.shp` + `dataset/osm/rn-comp-xa-190101-didi/edges.shp` + 卫星图，确认地理对齐 |
| 6 | 端到端 | `run_adamap.py` 用 `trajs=[]` 跑通，`fused_rn` 的边数应 > existing_rn 边数 |

## 常见问题

**Q：`ImportError: ... CXXABI_1.3.15 not found`？**
A：必须加 `LD_PRELOAD=/home/hanhaoyu/miniconda3/envs/SAM/lib/libstdc++.so.6`，或用 `conda install -c conda-forge libstdcxx-ng`。详见 [tptk/RUN.md:39-71](adaptive_map_completion/tptk/RUN.md#L39-L71)。

**Q：`AttributeError: module 'networkx' has no attribute 'read_shp'`？**
A：需要 `networkx<3`。`SAM` env 里是 2.8.8，OK。新建 env 用 `conda install networkx=2.8`。

**Q：拼图后空洞很多怎么办？**
A：检查 `images_full/` 下文件数是否为 2288；不足说明 `infer_all.py` 没跑完。

**Q：shp 文件巨大？**
A：调大 `--simplify_eps`（比如 4.0~6.0 像素），会更激进地化简曲线。

**Q：能否只对 test patches 拼图（不用 train patches）？**
A：不行，test 458 个 patch 像椒盐撒在城市里，不能拼出连贯路网。要做评测应在栅格层 per-patch 算 F1（即原 [test.py](Dual_Signal_Fusion_based_Map_Completion/test.py)），不要走"拼回大图再算"。
