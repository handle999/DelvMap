# didi_xian 2019sat 实验结果

> 2026-07-06。DSFNet 在 2019 Wayback sat 上的训练与评测结果(baseline)。
> sat 换 2019(release 645, 2019-06-26)以与 rn/traj 标签(2019)时间对齐,详见 samroad `docs/xian数据集2019影像替换与对齐.md`。

## 实验设定

| 项 | 值 |
|---|---|
| 模型 | DSFNet(双分支:traj 2ch + sat 3ch → road+building) |
| 数据集 | `dataset/didi_xian_train/`(零泄漏, 1512 patch, train 1208/val 148/test 156) |
| sat | 2019 Esri Wayback release 645(替换原 Google 2024) |
| 训练参数 | batch 16, lr 0.0005, lam 0.2, n_epochs 500(实际 228 early stop) |
| best 模型 | `checkpoints/didi_xian_2019sat/198_net_DSFNet.pth`(epoch 198) |
| split | 与 samroad 完全一致(test 39 region, 零泄漏) |

## 训练过程

- **best val F1 = 0.4144**(epoch 198),epoch 228 early stopping 触发(patience 用尽)。
- **严重过拟合**:train loss 0.35 / val loss 2.14;train F1 0.92/0.89/0.93 vs val F1 0.40/0.70/0.41。
- 训练日志:`checkpoints/didi_xian_2019sat/training.log`、`results.txt`(gitignore, 本地保留)。

## 评测结果

### Patch 级(`test.py`, test 156 patch, 256×256)

| 模态 | Precision | Recall | F1 | IOU |
|---|---|---|---|---|
| Traj(主路网) | 0.388 | 0.316 | **0.336** | 0.231 |
| Bldg(建筑) | 0.690 | 0.646 | **0.658** | 0.517 |
| Src(纯sat辅助) | 0.400 | 0.332 | **0.352** | 0.245 |

日志:`results/didi_xian_2019sat/test_198/test_log.txt`(gitignore, 本地保留)。

### Graph 级(`run_didi_eval.py`, test 35/39 region, AMC fused)

| 指标 | 值 | 说明 |
|---|---|---|
| **APLS** | **0.412** | 35/39 samples(4 region skip: partial 为空) |
| **TOPO F1** | **0.557** | |
| TOPO Precision | 0.687 | |
| TOPO Recall | 0.469 | |

结果:`didi_eval_2019sat/results/{apls.json, topo.json}` + per-region 明细(入库)。

## 对比基准

AMC baseline(2024 sat 时代, docs/评估指标与诊断-APLS-TOPO与碎片根因.md):

| 指标 | AMC baseline(2024 sat) | 2019sat(本实验) |
|---|---|---|
| APLS | 0.812 | 0.413 |
| TOPO F1 | 0.841 | 0.557 |
| TOPO R | 0.738 | 0.469 |

**退化主因**:2019 Wayback release 645 影像路网可见性弱(路处 vs 背景 sat 亮度差异仅 11-21,2024 高分影像更高),模型从 sat 学不到路(src F1 仅 0.352)。这是换 sat 的固有代价,非数据制作 bug——sat 与 gt/traj 坐标系对齐已验证(见 `docs/viz_data_check/`)。

## 复现命令

```bash
# 1. DSFNet 推断 (Mac MPS, 生成 images_full/)
python Dual_Signal_Fusion_based_Map_Completion/infer_all_mps.py \
    --name didi_xian_2019sat --dataroot ./dataset/didi_xian_train/ \
    --epoch 198 --model DSFNet --net_trans DSFNet --train_pattern DSFNet --gpu_ids -1

# 2. Graph 评测 (AMC 融合 + APLS/TOPO)
python adaptive_map_completion/run_didi_eval.py \
    --pred_dir results/didi_xian_2019sat/all_198/images_full \
    --split test --work_dir didi_eval_2019sat
```
