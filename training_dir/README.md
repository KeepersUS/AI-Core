# Grounding DINO Self-Training

Self-training setup for Grounding DINO object detection using pseudo-labels.

## Overview

This directory contains scripts for:
1. **Pseudo-label generation** - Using official Grounding DINO to generate labels for unlabeled images
2. **Model training** - Fine-tuning MMDetection's Grounding DINO on pseudo-labels
3. **Validation** - Checking label quality and training progress

## Directory Structure

```
training_dir/
├── scripts/              # Essential scripts
│   ├── train.py         # Main training script
│   ├── generate_pseudo_labels_official_weights.py  # Pseudo-label generation
│   ├── generate_full_dataset_official.sh           # Batch label generation
│   ├── visualize_official_labels.py                # Visualize labels
│   └── check_setup.py                               # Verify setup
├── data/
│   ├── annotations/     # COCO format annotations (pseudo-labels, val labels)
│   ├── unlabeled_images/ # Images to generate labels for
│   └── val/             # Validation images
├── checkpoints/         # Model checkpoints (gitignored)
├── work_dirs/           # Training outputs (gitignored)
└── mmdetection/         # MMDetection library

```

## Quick Start

### 1. Generate Pseudo-Labels

```bash
cd training_dir
./scripts/generate_full_dataset_official.sh
```

This generates labels in `data/annotations/pseudo_labels_official.json`.

### 2. Visualize Labels

```bash
python3 scripts/visualize_official_labels.py --num-samples 10
```

### 3. Train Model

```bash
python3 scripts/train.py
```

Training uses:
- **Config**: `mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py`
- **Checkpoint**: `checkpoints/groundingdino_swint_ogc.pth`
- **Labels**: `data/annotations/pseudo_labels_official.json`

## Key Files

- `TRAINING_MODEL_SETUP.md` - Detailed model architecture information
- `scripts/train.py` - Main training script
- `scripts/generate_pseudo_labels_official_weights.py` - Label generation
- `data/annotations/pseudo_labels_official.json` - Generated pseudo-labels

## Notes

- Pseudo-labels are generated using the **official Grounding DINO** implementation
- Training uses **MMDetection's Grounding DINO** (same architecture, different framework)
- See `.gitignore` for excluded files (visualizations, work_dirs, checkpoints, etc.)
