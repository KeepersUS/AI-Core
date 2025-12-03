#!/usr/bin/env python3
"""
Training script for Grounding DINO self-training.

This script handles model registration and runs training with proper initialization.
Works on both Mac (CPU) and Linux/Windows (GPU).

Usage:
    python scripts/train.py
    python scripts/train.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
MMDET_PATH = PROJECT_ROOT / 'mmdetection'

# Add mmdetection to path
sys.path.insert(0, str(MMDET_PATH))

# Change to mmdetection directory
os.chdir(MMDET_PATH)

# Handle device selection based on platform
import platform

if platform.system() == 'Darwin':  # macOS
    # Force CPU usage on Mac (MPS has issues with grid_sampler_2d_backward)
    # Set environment variable BEFORE importing torch - this is critical!
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    # Also set this to ensure CPU is used
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    print("⚠️  macOS detected - MPS fallback enabled")

import torch

# On Mac, explicitly disable MPS and force CPU
if platform.system() == 'Darwin':
    # Disable MPS backend
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Force CPU by monkey-patching is_available
        torch.backends.mps.is_available = lambda: False
    # Set default device to CPU
    torch.set_default_tensor_type('torch.FloatTensor')
    print("✅ CPU mode forced for Mac compatibility (MPS disabled)")
elif torch.cuda.is_available():
    print(f"✅ CUDA available - will use GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected - will use CPU (training will be slow)")

# CRITICAL: Import MMDetection models BEFORE anything else
# This ensures GroundingDINO is registered in the model registry
print("=" * 60)
print("Importing MMDetection Models")
print("=" * 60)
import mmdet
print(f"✅ MMDetection: {mmdet.__version__}")

# Explicitly import GroundingDINO to trigger registration
from mmdet.models import GroundingDINO
print("✅ GroundingDINO imported")

# Verify registration
from mmdet.registry import MODELS
try:
    model_cls = MODELS.get('GroundingDINO')
    print(f"✅ GroundingDINO registered: {model_cls}")
except KeyError:
    print("⚠️  GroundingDINO not in registry, but continuing...")
    # Force import of detectors module
    import mmdet.models.detectors
    print("✅ Detectors module imported")

print("=" * 60)
print()

# Use the base grounding_dino config directly with command-line overrides
# This avoids config inheritance issues
base_config = MMDET_PATH / 'configs' / 'grounding_dino' / 'grounding_dino_swin-t_finetune_16xb2_1x_coco.py'
work_dir = PROJECT_ROOT / 'work_dirs' / 'run_local'
checkpoint = PROJECT_ROOT / 'checkpoints' / 'groundingdino_swint_ogc.pth'
data_root = PROJECT_ROOT / 'data'

original_argv = sys.argv.copy()
sys.argv = [
    'train.py',
    str(base_config),
    '--work-dir', str(work_dir),
    '--cfg-options',
    f'load_from={checkpoint}',
    f'train_dataloader.batch_size=4',  # Can use larger batch on GPU
    f'train_dataloader.num_workers=4',  # More workers on Windows/GPU
    f'train_dataloader.dataset.data_root={data_root}',
    f'train_dataloader.dataset.data_prefix.img=unlabeled_images/',
    f'train_dataloader.dataset.ann_file=annotations/pseudo_labels_official.json',
    f'train_dataloader.dataset.filter_cfg.filter_empty_gt=False',
    f'train_dataloader.dataset.filter_cfg.min_size=32',
    f'val_dataloader.dataset.data_root={data_root}',
    f'val_dataloader.dataset.data_prefix.img=val/',
    f'val_dataloader.dataset.ann_file=annotations/val.json',
    f'test_dataloader.dataset.data_root={data_root}',
    f'test_dataloader.dataset.data_prefix.img=val/',
    f'test_dataloader.dataset.ann_file=annotations/val.json',
    f'val_evaluator.ann_file=../data/annotations/val.json',
    f'test_evaluator.ann_file=../data/annotations/val.json',
    f'val_evaluator.classwise=False',  # Disable classwise to avoid category ID issues
    f'val_dataloader.dataset.metainfo.classes=("bed", "book", "bowl", "chair", "couch", "dining table", "knife", "oven", "potted plant", "remote", "sink", "spoon", "toilet", "tv")',
    f'test_dataloader.dataset.metainfo.classes=("bed", "book", "bowl", "chair", "couch", "dining table", "knife", "oven", "potted plant", "remote", "sink", "spoon", "toilet", "tv")',
    f'model.bbox_head.num_classes=14',
    f'optim_wrapper.optimizer.lr=5e-6',  # Lower LR to prevent gradient explosion
    f'optim_wrapper.clip_grad.max_norm=0.5',  # Stronger gradient clipping
    f'train_cfg.max_epochs=12',  # More epochs for GPU training
    f'train_cfg.val_interval=1',  # Re-enable validation every epoch
    f'val_evaluator.classwise=False',  # Disable classwise metrics to avoid category ID issues
    f'default_hooks.logger.interval=10',
    f'env_cfg.mp_cfg.mp_start_method=fork',  # Use fork for Mac compatibility
    '--launcher', 'none'
]

try:
    # Import and call train.py's main function
    from tools.train import main as train_main
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)
    print(f"Base Config: {base_config}")
    print(f"Work dir: {work_dir}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Data root: {data_root}")
    print("=" * 60)
    print()
    
    train_main()
except SystemExit as e:
    sys.exit(e.code if hasattr(e, 'code') and e.code is not None else 0)
except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user")
    sys.exit(130)
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    sys.argv = original_argv

