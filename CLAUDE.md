# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DelvMap is a deep learning project for completing residential road maps using courier trajectories and satellite imagery. It has two stages:
1. **Dual Signal Fusion (DSFNet)** - Fuses trajectory data and satellite imagery to predict missing roads
2. **Adaptive Map Completion** - Post-processing to refine the road network

## Common Commands

### Training Stage 1 (DSFNet)
```bash
python Dual_Signal_Fusion_based_Map_Completion/train.py --name Your_Task_Name --dataroot Your_dataset --lam 0.2 --batch_size 8 --train_pattern DSFNet --net_trans DSFNet --model DSFNet
```

### Inference Stage 1
```bash
python Dual_Signal_Fusion_based_Map_Completion/test.py --name Your_Task_Name --dataroot Your_dataset --lam 0.2 --train_pattern DSFNet --net_trans DSFNet --model DSFNet --epoch XX
```

### Data Preparation
```bash
# Prepare raw OSM data (requires OSM PBF file)
python dataset/prepare_dataset.py

# Generate training patches from full-size images (random mode)
python dataset/create_dataset.py -i dataset/delvmap_data -o output -n 100

# Generate training patches (sliding mode with reproducible train/val/test split)
python dataset/create_dataset.py -i dataset/delvmap_data -o output -m sliding --stride 128

# Generate building masks
python dataset/generate_building_mask.py
```

### Data Split (sliding mode)
- Uses `split_indices.json` in dataset root for reproducible train/val/test split (80%/10%/10%, seed=42)
- Dataloader automatically reads from `split_indices.json`

## Architecture

### Data Flow
1. **Raw Data**: Satellite imagery (5625×6610) + Courier trajectories in WGS coordinates
2. **Preprocessing**: OSM PBF file → road/buildings → geo-to-pixel conversion
3. **Training Data**: Patches (256×256 typical) with 5 input channels:
   - `traj_data`: trajectory heatmap (path + points)
   - `src_data`: satellite image (RGB)
   - `label_data`: ground truth road mask
   - `building_label_data`: building mask

### Key Models
- **DSFNet** ([models/DSFNet_model.py](Dual_Signal_Fusion_based_Map_Completion/models/DSFNet_model.py)): Dual-branch UNet-style network
  - Input: 5-channel (traj 2ch + src 3ch)
  - Outputs: road prediction, building prediction, src-only road prediction
  - Losses: BCE + Dice loss combination
  - Metrics: Precision, Recall, F1, IOU

### Directory Structure
- `Dual_Signal_Fusion_based_Map_Completion/` - Stage 1 training/inference
  - `train.py` / `test.py` - Entry points
  - `models/` - DSFNet, losses, metrics
  - `data_loader.py` - Data loading pipeline
  - `tptk/` - Trajectory processing utilities (map matching, HMM)
- `adaptive_map_completion/` - Stage 2 refinement
- `dataset/` - Data preparation scripts
- `rawdata/` - Raw satellite imagery and trajectory data

### Configuration
- Geographic coverage: Xi'an, China (34.20°-34.28°N, 108.92°-108.99°E)
- Key config in [dataset/prepare_dataset.py](dataset/prepare_dataset.py): `CONFIG` dict with lat/lon bounds, image size (5625×6610), road width (2px)
