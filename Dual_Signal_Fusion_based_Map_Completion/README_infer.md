# DSFNet Inference Guide

## 1. Quick Start

### Basic Command

```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name delvmap_exp1 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --epoch 173 \
    --model DSFNet \
    --net_trans DSFNet \
    --train_pattern DSFNet \
    --gpu_ids 0
```

---

## 2. Parameter Details

### 2.1 Basic Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataroot` | Dataset root directory (required) | - |
| `--name` | Experiment name | `experiment_name` |
| `--gpu_ids` | GPU device ID | `0` |
| `--checkpoints_dir` | Checkpoint directory | `./checkpoints` |
| `--batch_size` | Batch size | `8` |
| `--num_threads` | Data loading threads | `4` |

### 2.2 Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model type | `Unet` |
| `--net_trans` | Network architecture | `Unet` |
| `--train_pattern` | Training mode | `simple` |
| `--output_nc` | Output channels | `1` |
| `--norm` | Normalization type | `batch` |
| `--init_type` | Weight initialization | `kaiming` |
| `--init_gain` | Initialization gain | `0.02` |

### 2.3 Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--epoch` | Model epoch to load | `latest` |
| `--load_iter` | Iteration to load | `0` |
| `--results_dir` | Results save directory | `./results/` |
| `--aspect_ratio` | Output image aspect ratio | `1.0` |
| `--phase` | Test phase | `test` |
| `--eval` | Use eval mode | False |

---

## 3. Input/Output Specification

### 3.1 Input Data (Test Set)

Loaded from test split:

| Data Type | Key | Shape | Description |
|-----------|-----|-------|-------------|
| Trajectory | `traj_data` | [B, 2, 256, 256] | 2-channel trajectory heatmap |
| Satellite | `src_data` | [B, 3, 256, 256] | RGB satellite image |
| Road Label | `label_data` | [B, 1, 256, 256] | Binary road mask |

### 3.2 Output Results

Saved to `results/{name}/test_{epoch}/`:

```
results/delvmap_exp1/test_173/
├── images/
│   ├── epoch173_*.png
│   └── ...
├── index.html
└── ...
```

**Output Image Types:**

| Filename | Description |
|----------|-------------|
| `*_pred_traj_img.png` | Road prediction (traj+satellite fusion) |
| `*_pred_building_img.png` | Building prediction |
| `*_pred_src_traj_img.png` | Road prediction (satellite only) |

---

## 4. Inference Pipeline

```
test.py
    |
    ├── 1. Load model
    |       └── checkpoints/{name}/{epoch}_net_DSFNet.pth
    |
    ├── 2. Load test set
    |       └── get_data_loader_multistage(dataroot, 'test')
    |
    ├── 3. Iterate each batch
    |       ├── set_input_test(data)
    |       ├── model.test()
    |       └── get_current_visuals()
    |
    └── 4. Save results
            └── results/{name}/test_{epoch}/
```

---

## 5. Dataset Structure

```
dataset_root/
├── split_indices.json
├── traj_and_point_split/     # Trajectory data (.npy)
├── label/                    # Road labels (.png)
├── src_split/                # Satellite images (.png)
└── building_label/           # Building labels (.png)
```

### split_indices.json Format

```json
{
    "train": [0, 1, ..., 799],
    "val": [800, 801, ..., 899],
    "test": [900, 901, ..., 999]
}
```

Default split: 80% train / 10% val / 10% test

---

## 6. Usage Examples

### Example 1: Test Best Model

```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name delvmap_exp1 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --epoch 173 \
    --model DSFNet \
    --net_trans DSFNet \
    --train_pattern DSFNet \
    --gpu_ids 0
```

### Example 2: Use Latest Model

```bash
--epoch latest
```

### Example 3: Custom Results Directory

```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name delvmap_exp1 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --epoch 173 \
    --results_dir ./my_results \
    --model DSFNet \
    --net_trans DSFNet \
    --train_pattern DSFNet
```

### Example 4: Val/Train Inference

```bash
--phase test    # Test set (default)
--phase val     # Validation set
--phase train   # Training set
```

### Example 5: Multi-GPU Inference

```bash
--gpu_ids 0,1,2,3
```

---

## 7. Output Metrics

After testing, prints:

```
loss_Dice    precision    recall    f1       iou
0.123456     0.8500       0.8200    0.8350   0.7200
```

**Metric Description:**
- `loss_Dice`: Dice Loss
- `precision`: Precision
- `recall`: Recall
- `f1`: F1 Score
- `iou`: Class IOU (road)

---

## 8. Choosing Prediction Output

DSFNet has three output heads:

| Output | File Suffix | Description | Use Case |
|--------|-------------|-------------|----------|
| pred_traj | `_pred_traj_img.png` | Traj+Sat fusion | With trajectory |
| pred_src_traj | `_pred_src_traj_img.png` | Satellite only | Pure vision (recommended) |
| pred_building | `_pred_building_img.png` | Building prediction | Building extraction |

**Recommended: `pred_src_traj_img` (satellite only)** because:
- No trajectory data needed
- Simpler deployment
- Matches training save criteria (F1)

---

## 9. Viewing Results

### Method 1: Open HTML Directly

```bash
firefox results/delvmap_exp1/test_173/index.html
```

### Method 2: HTTP Server

```bash
cd results/delvmap_exp1/test_173
python -m http.server 8000
# Browser: http://localhost:8000
```

---

## 10. Code Fixes

| Issue | Fix |
|-------|-----|
| Called non-existent `testtest()` method | Changed to `model.test()` |
| `model.get_current_visuals()` missing self | Changed to `self.model.get_current_visuals()` |
| `model.get_image_paths()` missing self | Changed to `self.model.get_image_paths()` |

---

## 11. Complete Parameter List

```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py \
    --name delvmap_exp1 \
    --dataroot ./dataset/xian_2019_delvmap/ \
    --epoch 173 \
    --model DSFNet \
    --net_trans DSFNet \
    --train_pattern DSFNet \
    --gpu_ids 0 \
    --results_dir ./results \
    --phase test \
    --batch_size 1 \
    --num_threads 4
```
