# STR Cleaning Verification — Object Detection Pipeline

RF-DETR-Large object detection system for short-term rental (STR) cleaning verification. Detects 40 classes of household objects in property photos to confirm that units have been properly cleaned and restocked between guest stays.

- **Model**: RF-DETR-Large (DINOv2 backbone, Apache 2.0 licensed)
- **Classes**: 40 (CL2), 640×480 images, COCO format
- **Target**: Macro-F1 > 0.85, no class below F1 = 0.65
- **Hardware**: RTX 5070 (12 GB VRAM), Ryzen 7 7800X3D, 32 GB DDR5

---

## Repository Layout

```
ai-dev/
├── README.md
├── requirements.txt
│
├── # ── Core scripts ────────────────────────────────────────────
├── train_rfdetr.py              # Train RF-DETR-Large
├── sweep_rfdetr_threshold.py    # Per-class threshold optimization
├── compare_all_models.py        # Evaluate / compare RF-DETR models
│
├── # ── Dataset assembly ────────────────────────────────────────
├── convert_to_coco.py           # Build COCO datasets from flat/firebase images
├── assemble_str_targeted.py     # Assemble targeted training set (Option A split)
├── migrate_coco_classes.py      # One-time CL1→CL2 category migration
├── check_dataset_overlap.py     # Safety check: overlap + coverage before assembly
├── find_annotation_candidates.py # Rank firebase properties by class richness
├── select_tier3_samples.py      # Stratified auto-label candidate sampler
│
├── # ── Utilities ───────────────────────────────────────────────
├── str_config.py                # Class list, label normalization, shared constants
├── str_visualizer.py            # Draw detection boxes on images (OpenCV)
├── resize_images.py             # Resize images to 640×480 with letterbox padding
│
├── # ── Weights ─────────────────────────────────────────────────
├── weights/
│   └── rf-detr-large-2026.pth   # Base pretrain weights (start CL2 training here)
│
├── # ── Trained model runs ──────────────────────────────────────
├── runs/
│   ├── str_rfdetr_martin_1/     # Best CL1 model (Martin, 859 imgs, 100 ep)
│   └── str_rfdetr_martin_v2c/   # Martin fine-tune variant (CL1)
│
├── # ── Datasets ────────────────────────────────────────────────
├── datasets/
│   ├── CL1/                     # CL1 (30-class) reference datasets
│   │   ├── str_coco_martin_1/   # Martin base training set (859 images)
│   │   └── str_coco_martin_v2d/ # Martin v2d fine-tune set
│   └── CL2/                     # CL2 (40-class) datasets — active
│       └── str_coco_martin_new_CL/
│           ├── train/           # 687 images, 4 659 annotations
│           ├── valid/           # 171 images, 1 157 annotations
│           └── test/            # Empty COCO JSON (required by RF-DETR)
│
├── # ── Results / outputs ───────────────────────────────────────
├── results/
│   ├── thresholds/              # Per-class threshold JSONs from sweep
│   │   ├── per_class_thresholds_RF-DETR-Martin.json
│   │   └── per_class_thresholds_Martin-v2c.json
│   ├── comparisons/             # JSON outputs from compare_all_models.py
│   ├── evaluation/              # Misc evaluation outputs
│   └── annotation_candidates/  # CSVs from find_annotation_candidates.py
│
├── # ── Test photos / validation ────────────────────────────────
├── test_photos/
│   ├── firebase/
│   │   ├── annotator_improved.py        # Manual annotation GUI (active annotator)
│   │   └── properties/                  # Firebase property image cache
│   ├── properties/                      # Co-op annotation image store (offline access)
│   ├── validation_CL2/
│   │   ├── validation_new_images_CL2/   # 224 held-out validation images
│   │   └── validation_new_labels_CL2/   # Co-located JSON labels
│   └── hand_annotated_CL2/             # Hand-annotated images staged for conversion
│
├── # ── Tools ───────────────────────────────────────────────────
└── tools/
    └── annotation-tool/         # Git submodule — web annotation UI
```

---

## Class List — CL2 (40 classes)

| ID | Class | Notes |
|---|---|---|
| 0 | bed | |
| 1 | pillow | |
| 2 | couch | includes outdoor variants |
| 3 | chair | includes stool, bench |
| 4 | table | |
| 5 | blanket | comforter, duvet |
| 6 | television | |
| 7 | lamp | |
| 8 | mirror | |
| 9 | curtain | includes shower curtain |
| 10 | rug | |
| 11 | fan | |
| 12 | sink | |
| 13 | refrigerator | |
| 14 | coffee maker | |
| 15 | remote | |
| 16 | basket | laundry basket |
| 17 | paper towel roll | |
| 18 | stove | |
| 19 | oven | |
| 20 | microwave | |
| 21 | dishwasher | |
| 22 | toilet | |
| 23 | shower | includes bathtub |
| 24 | towel | hand towels, washrags |
| 25 | toilet paper roll | |
| 26 | trashcan | |
| 27 | decor | replaces `picture` (CL1); superset: art, books, wall decor |
| 28 | washer/dryer | |
| 29 | plant | |
| 30 | soap | dish/hand soap, shampoo, cleaning liquids |
| 31 | drinkwear | cups, glasses, mugs |
| 32 | plunger | plunger, toilet brush |
| 33 | tv stand | media console, entertainment center |
| 34 | dresser | chest-of-drawers, sideboard, bureau |
| 35 | nightstand | bedside table |
| 36 | toaster | |
| 37 | coat rack | hat rack, clothes rack |
| 38 | coffee pods | k-cups, nespresso pods |
| 39 | broom | mop, pool vacuum |

IDs 30–39 are new in CL2 and currently have **zero annotations** — annotation sessions are required before training a CL2 model. IDs 35–39 are "questionable" and may be merged into broader classes if F1 is poor after first CL2 training.

---

## Models

All best weights live at `runs/<run>/checkpoint_best_ema.pth`.

| Run | Label | Dataset | Classes | Epochs | Status |
|---|---|---|---|---|---|
| `str_rfdetr_martin_1` | Martin | str_coco_martin_1 (859 train) | CL1 (30) | 100 | Best CL1 overall |
| `str_rfdetr_martin_v2c` | Martin-v2c | str_coco_martin_v2d | CL1 (30) | — | Fine-tune variant |

**CL2 models not yet trained.** The class count changed (30→40), so CL1 checkpoints cannot be fine-tuned. All CL2 training must start from `rf-detr-large-2026.pth`.

### CL1 Evaluation Results (held-out val: 224 images / 1 363 GT objects)

| Model | Precision | Recall | Micro-F1 | Macro-F1 |
|---|---|---|---|---|
| Martin | 0.8568 | 0.8298 | 0.8431 | **0.8491** |
| Targeted-v2 | 0.8618 | 0.8100 | 0.8351 | 0.8390 |

Martin is the current best CL1 model.

---

## Common Commands

### Training

```bash
# Train from base weights (required for CL2 — class count changed)
python train_rfdetr.py \
  --data datasets/CL2/<dataset> \
  --model large \
  --output runs/<new_run> \
  --epochs 200 --batch 8 --workers 0 --grad-accum 1

# Fine-tune from an existing CL2 checkpoint
python train_rfdetr.py \
  --data datasets/CL2/<dataset> \
  --model large \
  --resume runs/<checkpoint>/checkpoint_best_ema.pth \
  --output runs/<new_run> \
  --epochs 75 --batch 8 --workers 0 --grad-accum 1 --lr 1e-5
```

> **Batch size note**: `--batch 8` is the safe limit for 12 GB VRAM. Batch 16 exceeds VRAM and spills to PCIe. Always use `--workers 0` on Windows.

### Threshold Sweep

```bash
python sweep_rfdetr_threshold.py \
  --new-val --per-class \
  --weights runs/<run>/checkpoint_best_ema.pth \
  --label <ModelLabel>
# Output: results/thresholds/per_class_thresholds_<ModelLabel>.json
```

### Model Comparison

```bash
python compare_all_models.py \
  --new-val --skip-dino --skip-rt \
  --rfdetr-weights runs/<run1>/checkpoint_best_ema.pth \
  --rfdetr-label "<Label1>" \
  --rfdetr-per-class-thresh results/thresholds/per_class_thresholds_<Label1>.json \
  --rfdetr-weights2 runs/<run2>/checkpoint_best_ema.pth \
  --rfdetr-label2 "<Label2>" \
  --rfdetr-per-class-thresh2 results/thresholds/per_class_thresholds_<Label2>.json
```

`--skip-dino` and `--skip-rt` are always required (those models are deprecated).

### Annotation

**Annotate training images** (existing COCO boxes pre-loaded):
```bash
py test_photos/firebase/annotator_improved.py \
  --images datasets_CL2/str_coco_martin_new_CL/train \
  --dataset datasets/CL2/str_coco_martin_new_CL \
  --hand-annotated-dir test_photos/hand_annotated_CL2 \
  --firebase-root datasets/CL2/str_coco_martin_new_CL/train \
  --coco-annotations datasets/CL2/str_coco_martin_new_CL/train/_annotations.coco.json
```

**Validation expansion mode** (add images to the held-out val set):
```bash
py test_photos/firebase/annotator_improved.py \
  --images <source_dir> \
  --dataset datasets/CL2/str_coco_martin_new_CL \
  --hand-annotated-dir test_photos/hand_annotated_CL2 \
  --firebase-root test_photos/firebase/properties \
  --validation-mode \
  --val-images-dir test_photos/validation_CL2/validation_new_images_CL2 \
  --val-labels-dir test_photos/validation_CL2/validation_new_labels_CL2
```

Co-located JSONs saved by the annotator always take precedence over `--coco-annotations` on subsequent loads.

### Dataset Assembly

```bash
# Build / rebuild a COCO dataset from flat or firebase-layout images
python convert_to_coco.py [options]

# Assemble the targeted training set (Option A split strategy)
python assemble_str_targeted.py

# Find the best firebase properties to annotate next
python find_annotation_candidates.py
```

---

## File Conventions

| Convention | Detail |
|---|---|
| COCO val split folder | Must be named `valid` (not `val`) — RF-DETR requirement |
| Test split | Must exist as an empty COCO JSON to avoid `FileNotFoundError` at training start |
| Firebase co-located labels | JSON beside the image, same stem — format: `[{"class": "...", "bbox": [x1,y1,x2,y2]}]` |
| Image size | 640×480 — use `resize_images.py` to normalize before dataset assembly |
| Label normalization | All label variants (e.g. `"tv"`, `"sofa"`) are mapped to canonical names via `str_config.normalize_label()` |

---

## Next Steps

1. **Annotate new CL2 classes** — run `annotator_improved.py` with `--coco-annotations`; target 50–100 instances per new class minimum (IDs 30–39 currently have 0 annotations)
2. **Rebuild the CL2 COCO dataset** — merge existing CL2 COCO JSON with new hand-annotated co-located JSONs from `test_photos/hand_annotated_CL2/`
3. **Train first CL2 model** from `rf-detr-large-2026.pth`
4. **Threshold sweep** on the new model, then evaluate against `validation_CL2/`
5. **Evaluate questionable classes** (IDs 35–39) — merge into broader class if F1 is poor

---

## Tools

### annotation-tool (git submodule)

`tools/annotation-tool` is a git submodule pointing to an external web-based annotation UI.

**First-time setup** (after cloning this repo):
```bash
git submodule update --init tools/annotation-tool
```

**Pull latest changes from the submodule's upstream repo:**
```bash
git submodule update --remote tools/annotation-tool
```

After updating, commit the new submodule pointer so others get the same revision:
```bash
git add tools/annotation-tool
git commit -m "Update annotation-tool submodule"
```

**Check which commit the submodule is pinned to:**
```bash
git submodule status
```
