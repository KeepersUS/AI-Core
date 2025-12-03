# Training Model Setup

## Summary

You'll be using **MMDetection's Grounding DINO model** for training, not the official Grounding DINO implementation.

## Model Usage Breakdown

### 1. **Pseudo-Label Generation** (✅ Already Done)
- **Model Used**: Official Grounding DINO (`groundingdino` package)
- **Script**: `generate_pseudo_labels_official_weights.py`
- **Why**: To generate high-quality pseudo-labels with correct bounding box placements
- **Weights**: `weights/groundingdino_swint_ogc.pth`

### 2. **Training** (Next Step)
- **Model Used**: MMDetection's Grounding DINO implementation
- **Script**: `scripts/train.py`
- **Config**: `mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py`
- **Base Model**: Swin-T (Swin Transformer Tiny) backbone
- **Why**: MMDetection provides training infrastructure, checkpoints, and compatibility

## Model Architecture Details

Both use the same underlying architecture:
- **Backbone**: Swin Transformer (Swin-T)
- **Architecture**: Grounding DINO (Transformer-based detector)
- **Text Encoder**: BERT-base-uncased
- **Features**: Open-vocabulary object detection

The difference is just the implementation framework:
- **Official**: Direct from IDEA Research (used for inference/pseudo-labeling)
- **MMDetection**: Wrapped for training with MMDetection's training framework

## Training Configuration

Your training script (`scripts/train.py`) will:
1. Load MMDetection's Grounding DINO config
2. Initialize from checkpoint: `checkpoints/groundingdino_swint_ogc.pth`
3. Fine-tune on your pseudo-labels: `data/annotations/pseudo_labels_official.json`
4. Validate on: `data/annotations/val.json`

## Key Files

- **Training Config**: `mmdetection/configs/grounding_dino/grounding_dino_swin-t_finetune_16xb2_1x_coco.py`
- **Training Script**: `scripts/train.py`
- **Pseudo-Labels**: `data/annotations/pseudo_labels_official.json`
- **Checkpoint**: `checkpoints/groundingdino_swint_ogc.pth`

## Next Steps

1. ✅ Pseudo-labels generated (using official DINO)
2. ✅ Quality verified (visualizations look good)
3. 🔄 Ready to train (using MMDetection DINO)

To start training:
```bash
cd training_dir
python scripts/train.py
```

Or update the config to point to your new pseudo-labels file:
- Change line 98 in `scripts/train.py`:
  - From: `f'train_dataloader.dataset.ann_file=annotations/pseudo_labels_v1.json'`
  - To: `f'train_dataloader.dataset.ann_file=annotations/pseudo_labels_official.json'`

