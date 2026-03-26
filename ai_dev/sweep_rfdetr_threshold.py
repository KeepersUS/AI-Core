"""
RF-DETR Confidence Threshold Sweep

Find the optimal confidence threshold for RF-DETR to maximize F1 score.

Usage:
    python sweep_rfdetr_threshold.py              # COCO test set
    python sweep_rfdetr_threshold.py --new-val    # New validation set (held-out)
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import torch
from PIL import Image
from torchvision import transforms

from str_config import STR_CORE_CLASSES, normalize_label

# Default paths (overridable via CLI)
RFDETR_WEIGHTS = Path("runs/str_rfdetr_v3b/checkpoint_best_ema.pth")
DATASET_DIR = Path("datasets/str_coco")


def load_coco_annotations(ann_file: Path):
    """Load COCO format annotations."""
    with open(ann_file) as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    return images, annotations_by_image, categories


def load_individual_annotations(images_dir: Path, labels_dir: Path, class_list: List[str] = None):
    """Load individual JSON annotations (one JSON per image)."""
    if class_list is None:
        class_list = STR_CORE_CLASSES
    categories = {i: cls for i, cls in enumerate(class_list)}
    class_to_id = {cls: i for i, cls in enumerate(class_list)}
    
    images = {}
    annotations_by_image = defaultdict(list)
    
    img_id = 0
    matched = 0
    
    for img_file in sorted(images_dir.glob("*.jpg")):
        label_file = labels_dir / f"{img_file.stem}.json"
        if not label_file.exists():
            continue
        
        img_id += 1
        images[img_id] = {"id": img_id, "file_name": img_file.name}
        
        with open(label_file) as f:
            anns = json.load(f)
        
        for ann in anns:
            cls = ann["class"]
            if cls not in class_to_id:
                continue
            x1, y1, x2, y2 = ann["bbox"]
            annotations_by_image[img_id].append({
                "category_id": class_to_id[cls],
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # Convert to COCO [x,y,w,h]
            })
        matched += 1
    
    total_anns = sum(len(v) for v in annotations_by_image.values())
    print(f"  Loaded {matched} images, {total_anns} ground truth objects")
    return images, annotations_by_image, categories


def coco_bbox_to_xyxy(bbox: List[float]) -> List[float]:
    """Convert COCO [x,y,w,h] to [x1,y1,x2,y2]."""
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def evaluate_predictions(predictions: List[Dict], ground_truth: List[Dict], iou_threshold: float = 0.5):
    """Match predictions to ground truth and return metrics."""
    matched_gt = set()
    tp = 0
    fp = 0
    
    for pred in predictions:
        pred_class = pred["class"]
        pred_box = pred["bbox"]
        
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if pred_class != gt["class"]:
                continue
            iou = calculate_iou(pred_box, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def load_rfdetr_model(weights_path: Path = None):
    """Load RF-DETR model with fine-tuned weights."""
    weights = weights_path or RFDETR_WEIGHTS
    print(f"Loading RF-DETR model from: {weights}")
    
    # Load checkpoint
    checkpoint = torch.load(str(weights), map_location='cpu', weights_only=False)
    
    # Get state dict (prefer EMA weights)
    if 'model_ema' in checkpoint:
        state_dict = checkpoint['model_ema']
        print("  Using EMA weights")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("  Using model weights")
    else:
        state_dict = checkpoint
    
    # Detect num_classes from checkpoint
    num_classes = state_dict['class_embed.weight'].shape[0]
    print(f"  Detected {num_classes} classes from checkpoint")
    
    # Load RF-DETR and resize classification head
    from rfdetr import RFDETRLarge
    import sys, io
    
    # Suppress output during model creation
    old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        model = RFDETRLarge()
    finally:
        sys.stdout = old_stdout
    
    # Get the internal model
    if hasattr(model.model, 'model'):
        torch_model = model.model.model
    else:
        torch_model = model.model
    
    # Resize classification layers to match checkpoint
    for name, param in state_dict.items():
        if 'class_embed' in name or 'enc_out_class_embed' in name:
            parts = name.split('.')
            obj = torch_model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            target_param = getattr(obj, parts[-1])
            if target_param.shape != param.shape:
                setattr(obj, parts[-1], torch.nn.Parameter(param.clone()))
    
    # Load full state dict
    torch_model.load_state_dict(state_dict, strict=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_model.to(device)
    torch_model.eval()
    print(f"  ✓ Model loaded on {device}")
    
    return torch_model, num_classes, device


def run_threshold_sweep(use_new_val: bool = False, weights_path: Path = None,
                        model_label: str = "RF-DETR", fine: bool = False,
                        class_list: List[str] = None,
                        val_images: Path = None, val_labels: Path = None):
    """Sweep confidence thresholds to find optimal settings."""
    if class_list is None:
        class_list = STR_CORE_CLASSES

    dataset_label = "New Validation Set (held-out)" if use_new_val else "COCO Test Set"

    print("=" * 70)
    print(f"RF-DETR CONFIDENCE THRESHOLD SWEEP — {model_label}")
    print(f"Dataset: {dataset_label}")
    print("=" * 70)

    # Load model
    torch_model, num_classes, device = load_rfdetr_model(weights_path)
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    if val_images and val_labels:
        if not val_images.exists() or not val_labels.exists():
            print(f"Validation set not found: {val_images} / {val_labels}")
            return
        images, annotations, categories = load_individual_annotations(val_images, val_labels, class_list)
        test_imgs_dir = val_images
    elif use_new_val:
        new_val_images = Path("test_photos/validation_CL2/validation_new_images_CL2")
        new_val_labels = Path("test_photos/validation_CL2/validation_new_labels_CL2")
        if not new_val_images.exists() or not new_val_labels.exists():
            print(f"New validation set not found")
            return
        images, annotations, categories = load_individual_annotations(new_val_images, new_val_labels, class_list)
        test_imgs_dir = new_val_images
    else:
        test_ann = DATASET_DIR / "test" / "_annotations.coco.json"
        test_imgs_dir = DATASET_DIR / "test"
        if not test_ann.exists():
            print(f"❌ Test annotations not found: {test_ann}")
            return
        images, annotations, categories = load_coco_annotations(test_ann)

    print(f"Evaluation images: {len(images)}")
    
    # Broad sweep by default to locate the peak for an unseen model.
    # Use --fine to zoom in around the 0.85-0.95 region once peak is known.
    if fine:
        thresholds = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95]
    else:
        thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.83, 0.86, 0.89, 0.92, 0.95, 0.97]
    
    results = []
    
    for conf_threshold in thresholds:
        print(f"\n--- Testing conf={conf_threshold:.2f} ---")
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Per-class tracking
        class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        
        for i, (img_id, img_info) in enumerate(images.items()):
            print(f"\r  Processing: {i+1}/{len(images)}", end="", flush=True)
            
            img_path = test_imgs_dir / img_info['file_name']
            if not img_path.exists():
                continue
            
            # Get ground truth
            gt_annotations = annotations.get(img_id, [])
            ground_truth = []
            for ann in gt_annotations:
                cat_name = categories.get(ann['category_id'], '')
                if cat_name in class_list:
                    ground_truth.append({
                        'class': cat_name,
                        'bbox': coco_bbox_to_xyxy(ann['bbox'])
                    })

            # Run inference
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = torch_model(img_tensor)

            # Process outputs
            pred_logits = outputs['pred_logits'][0]
            pred_boxes = outputs['pred_boxes'][0]

            probs = pred_logits.softmax(-1)
            scores, labels = probs.max(-1)

            predictions = []
            for score, label, box in zip(scores, labels, pred_boxes):
                score_val = score.item()
                label_val = label.item()

                if score_val < conf_threshold:
                    continue
                if label_val >= num_classes or label_val >= len(class_list):
                    continue

                cx, cy, w, h = box.tolist()
                x1 = int((cx - w/2) * orig_w)
                y1 = int((cy - h/2) * orig_h)
                x2 = int((cx + w/2) * orig_w)
                y2 = int((cy + h/2) * orig_h)

                predictions.append({
                    'class': class_list[label_val],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score_val
                })
            
            # Evaluate
            tp, fp, fn = evaluate_predictions(predictions, ground_truth)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            # Per-class evaluation
            gt_by_class = defaultdict(list)
            for gt in ground_truth:
                gt_by_class[gt["class"]].append(gt)
            
            pred_by_class = defaultdict(list)
            for pred in predictions:
                pred_by_class[pred["class"]].append(pred)
            
            all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())
            for cls in all_classes:
                tp_c, fp_c, fn_c = evaluate_predictions(
                    pred_by_class.get(cls, []),
                    gt_by_class.get(cls, [])
                )
                class_stats[cls]["tp"] += tp_c
                class_stats[cls]["fp"] += fp_c
                class_stats[cls]["fn"] += fn_c
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Macro metrics
        class_f1s = []
        for cls, stats in class_stats.items():
            p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
            r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if (stats["tp"] + stats["fn"]) > 0:
                class_f1s.append(f)
        
        macro_f1 = sum(class_f1s) / len(class_f1s) if class_f1s else 0
        
        results.append({
            "threshold": conf_threshold,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "micro_f1": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        })
        
        print(f"\n  Precision: {precision:.4f}, Recall: {recall:.4f}, Micro-F1: {f1:.4f}, Macro-F1: {macro_f1:.4f}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 70)
    print(f"{'Threshold':>10} {'Precision':>12} {'Recall':>12} {'Micro-F1':>12} {'Macro-F1':>12}")
    print("-" * 60)
    
    best_micro = max(results, key=lambda x: x["micro_f1"])
    best_macro = max(results, key=lambda x: x["macro_f1"])
    best_precision = max(results, key=lambda x: x["precision"])
    best_recall = max(results, key=lambda x: x["recall"])
    
    for r in results:
        markers = []
        if r == best_micro:
            markers.append("★ BEST-F1")
        if r == best_precision:
            markers.append("best-P")
        if r == best_recall:
            markers.append("best-R")
        
        marker_str = f" <- {', '.join(markers)}" if markers else ""
        
        print(f"{r['threshold']:>10.2f} {r['precision']:>12.4f} {r['recall']:>12.4f} "
              f"{r['micro_f1']:>12.4f} {r['macro_f1']:>12.4f}{marker_str}")
    
    print("\n" + "-" * 70)
    print("OPTIMAL THRESHOLDS")
    print("-" * 70)
    print(f"  Best Micro-F1:  {best_micro['threshold']:.2f} (F1={best_micro['micro_f1']:.4f}, P={best_micro['precision']:.4f}, R={best_micro['recall']:.4f})")
    print(f"  Best Precision: {best_precision['threshold']:.2f} (P={best_precision['precision']:.4f})")
    print(f"  Best Recall:    {best_recall['threshold']:.2f} (R={best_recall['recall']:.4f})")
    
    # Save results
    output_dir = Path("results/thresholds")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "rfdetr_sweep_results.json", 'w') as f:
        json.dump({
            "results": results,
            "best_micro_f1": best_micro,
            "best_precision": best_precision,
            "best_recall": best_recall,
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/rfdetr_sweep_results.json")
    
    # Show the curve trend
    print("\n" + "-" * 70)
    print("F1 TREND")
    print("-" * 70)
    
    max_f1 = max(r["micro_f1"] for r in results)
    
    for r in results:
        bar_len = int(r["micro_f1"] * 50)
        bar = "█" * bar_len
        marker = " ★ BEST" if r["micro_f1"] == max_f1 else ""
        print(f"  {r['threshold']:.2f}: {bar} {r['micro_f1']:.4f}{marker}")
    
    # Comparison with default 0.50 baseline
    baseline = next((r for r in results if r["threshold"] == 0.50), results[0])
    baseline_f1 = baseline["micro_f1"]
    improvement = best_micro["micro_f1"] - baseline_f1
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Default threshold (0.50):  Micro-F1 = {baseline_f1:.4f}")
    print(f"  Optimal threshold ({best_micro['threshold']:.2f}): Micro-F1 = {best_micro['micro_f1']:.4f}")
    if baseline_f1 > 0:
        print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_f1*100:+.2f}%)")
    else:
        print(f"  Improvement: {improvement:+.4f}")


def run_per_class_threshold_sweep(
    use_new_val: bool = False,
    weights_path: Path = None,
    model_label: str = "RF-DETR",
    global_conf: float = 0.89,
    min_conf: float = 0.500,
    max_conf: float = 0.975,
    step: float = 0.025,
    gt_floor_count: int = 20,
    gt_floor_thresh: float = 0.50,
    class_list: List[str] = None,
    val_images: Path = None,
    val_labels: Path = None,
):
    """
    Find the optimal confidence threshold independently for every class.

    Strategy:
      1. Single inference pass over the validation set — all predictions are
         stored without any confidence filter.
      2. Post-processing sweep: 20 thresholds (0.500 → 0.975, step 0.025) are
         applied per-class without re-running the model.
      3. The threshold that maximises F1 for each class is chosen.  When tied,
         the lower threshold wins (more recall-friendly).

    Output:
      results/thresholds/per_class_thresholds_<label>.json  — ready to load at
      inference time as a {class_name: threshold} map.
    """
    if class_list is None:
        class_list = STR_CORE_CLASSES

    dataset_label = "New Validation Set (held-out)" if use_new_val else "COCO Test Set"

    print("=" * 75)
    print(f"RF-DETR PER-CLASS THRESHOLD SWEEP — {model_label}")
    print(f"Dataset:          {dataset_label}")
    print(f"Global baseline:  {global_conf:.3f}")
    print("=" * 75)

    torch_model, num_classes, device = load_rfdetr_model(weights_path)

    preprocess = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    if val_images and val_labels:
        if not val_images.exists() or not val_labels.exists():
            print(f"Validation set not found: {val_images} / {val_labels}")
            return
        images, annotations, categories = load_individual_annotations(val_images, val_labels, class_list)
        test_imgs_dir = val_images
    elif use_new_val:
        new_val_images = Path("test_photos/validation_CL2/validation_new_images_CL2")
        new_val_labels = Path("test_photos/validation_CL2/validation_new_labels_CL2")
        if not new_val_images.exists() or not new_val_labels.exists():
            print("New validation set not found.")
            return
        images, annotations, categories = load_individual_annotations(new_val_images, new_val_labels, class_list)
        test_imgs_dir = new_val_images
    else:
        test_ann = DATASET_DIR / "test" / "_annotations.coco.json"
        test_imgs_dir = DATASET_DIR / "test"
        if not test_ann.exists():
            print(f"Test annotations not found: {test_ann}")
            return
        images, annotations, categories = load_coco_annotations(test_ann)

    print(f"Evaluation images: {len(images)}")

    # -------------------------------------------------------------------------
    # SINGLE INFERENCE PASS — no confidence filter, collect everything
    # -------------------------------------------------------------------------
    print(f"\nRunning inference (single pass)...")

    all_predictions: Dict[int, List[Dict]] = {}
    all_ground_truth: Dict[int, List[Dict]] = {}

    for i, (img_id, img_info) in enumerate(images.items(), 1):
        print(f"\r  {i}/{len(images)}", end="", flush=True)

        img_path = test_imgs_dir / img_info['file_name']
        if not img_path.exists():
            all_predictions[img_id] = []
            all_ground_truth[img_id] = []
            continue

        # Ground truth
        gt_anns = annotations.get(img_id, [])
        ground_truth = []
        for ann in gt_anns:
            cat_name = categories.get(ann['category_id'], '')
            if cat_name in class_list:
                ground_truth.append({
                    'class': cat_name,
                    'bbox': coco_bbox_to_xyxy(ann['bbox'])
                })
        all_ground_truth[img_id] = ground_truth

        # Inference — store all queries, no threshold
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = torch_model(img_tensor)

        pred_logits = outputs['pred_logits'][0]
        pred_boxes  = outputs['pred_boxes'][0]
        probs = pred_logits.softmax(-1)
        scores, labels = probs.max(-1)

        predictions = []
        for score, label, box in zip(scores, labels, pred_boxes):
            score_val = score.item()
            label_val = label.item()
            if label_val >= num_classes or label_val >= len(class_list):
                continue
            cx, cy, w, h = box.tolist()
            x1 = int((cx - w / 2) * orig_w)
            y1 = int((cy - h / 2) * orig_h)
            x2 = int((cx + w / 2) * orig_w)
            y2 = int((cy + h / 2) * orig_h)
            predictions.append({
                'class': class_list[label_val],
                'bbox': [x1, y1, x2, y2],
                'confidence': score_val,
            })
        all_predictions[img_id] = predictions

    print(f"\n  Done — {sum(len(v) for v in all_predictions.values())} raw predictions stored.")

    # -------------------------------------------------------------------------
    # THRESHOLD SWEEP — driven by min_conf/max_conf/step CLI args
    # All classes swept simultaneously per threshold step (one loop).
    # -------------------------------------------------------------------------
    n_steps = round((max_conf - min_conf) / step) + 1
    thresholds = [round(min_conf + i * step, 3) for i in range(n_steps)]

    print(f"\nSweeping {len(thresholds)} thresholds × {len(class_list)} classes "
          f"({thresholds[0]:.3f} → {thresholds[-1]:.3f}, step {step})...")

    # class_sweep[cls] = list of result dicts in threshold order
    class_sweep: Dict[str, List[Dict]] = {cls: [] for cls in class_list}

    # Ground truth count per class (used for display)
    gt_counts: Dict[str, int] = defaultdict(int)
    for img_id in images:
        for gt in all_ground_truth.get(img_id, []):
            gt_counts[gt['class']] += 1

    for threshold in thresholds:
        for cls in class_list:
            total_tp = total_fp = total_fn = 0
            for img_id in images:
                preds = [p for p in all_predictions.get(img_id, [])
                         if p['class'] == cls and p['confidence'] >= threshold]
                gts   = [g for g in all_ground_truth.get(img_id, [])
                         if g['class'] == cls]
                tp, fp, fn = evaluate_predictions(preds, gts)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            p  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            class_sweep[cls].append({
                'threshold': threshold,
                'precision': round(p, 4),
                'recall':    round(r, 4),
                'f1':        round(f1, 4),
                'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            })

    # -------------------------------------------------------------------------
    # OPTIMAL THRESHOLD PER CLASS
    # Ties broken by preferring the lower threshold (higher recall).
    # -------------------------------------------------------------------------
    optimal: Dict[str, Dict] = {}
    for cls in class_list:
        # max() is stable in Python — iterating low→high means the first
        # (lowest) threshold wins any tie when we invert and use min.
        best_f1 = max(r['f1'] for r in class_sweep[cls])
        # Among all entries matching best_f1, take the one with lowest threshold
        best = next(r for r in class_sweep[cls] if r['f1'] == best_f1)
        optimal[cls] = best

    # -------------------------------------------------------------------------
    # GT FLOOR — clamp per-class thresholds for under-represented classes
    # -------------------------------------------------------------------------
    floored_classes = []
    for cls in class_list:
        if gt_counts.get(cls, 0) < gt_floor_count and optimal[cls]['threshold'] < gt_floor_thresh:
            # Pick the lowest threshold >= gt_floor_thresh from the sweep
            floor_result = next(
                (r for r in class_sweep[cls] if r['threshold'] >= gt_floor_thresh), None
            )
            if floor_result:
                optimal[cls] = floor_result
                floored_classes.append(cls)

    if floored_classes:
        print(f"\nApplied {gt_floor_thresh:.2f} floor to {len(floored_classes)} classes "
              f"(GT < {gt_floor_count}): {', '.join(floored_classes)}")

    # F1 at the global baseline threshold for comparison
    global_f1_by_class: Dict[str, float] = {}
    for cls in class_list:
        closest = min(class_sweep[cls], key=lambda x: abs(x['threshold'] - global_conf))
        global_f1_by_class[cls] = closest['f1']

    # -------------------------------------------------------------------------
    # RESULTS TABLE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"PER-CLASS RESULTS  (baseline = global conf {global_conf:.3f})")
    print("=" * 75)
    print(f"{'Class':<20} {'GT':>4}  {'Global F1':>9}  {'Opt Thresh':>10}  {'Opt F1':>8}  {'Delta':>7}")
    print("-" * 75)

    total_global = total_opt = n_with_gt = 0

    for cls in class_list:
        gt_n  = gt_counts.get(cls, 0)
        g_f1  = global_f1_by_class[cls]
        opt   = optimal[cls]
        delta = opt['f1'] - g_f1

        if gt_n > 0:
            total_global += g_f1
            total_opt    += opt['f1']
            n_with_gt    += 1

        marker = " **" if delta >= 0.05 else (" *" if delta >= 0.02 else "")
        print(f"{cls:<20} {gt_n:>4}  {g_f1:>9.4f}  {opt['threshold']:>10.3f}  "
              f"{opt['f1']:>8.4f}  {delta:>+7.4f}{marker}")

    print("-" * 75)
    macro_global = total_global / n_with_gt if n_with_gt else 0
    macro_opt    = total_opt    / n_with_gt if n_with_gt else 0
    print(f"{'MACRO AVERAGE':<20} {'':>4}  {macro_global:>9.4f}  {'—':>10}  "
          f"{macro_opt:>8.4f}  {macro_opt - macro_global:>+7.4f}")
    print("  ** >= 0.05 improvement    * >= 0.02 improvement")

    # -------------------------------------------------------------------------
    # F1 CURVES — top 5 most improved classes
    # -------------------------------------------------------------------------
    improvements = [
        (cls, optimal[cls]['f1'] - global_f1_by_class[cls])
        for cls in class_list if gt_counts.get(cls, 0) > 0
    ]
    improvements.sort(key=lambda x: -x[1])

    thresh_header = " ".join(f"{t:.2f}" for t in thresholds)
    col_w = len(thresh_header)

    print(f"\n{'-' * 75}")
    print("F1 CURVES — TOP 5 MOST IMPROVED CLASSES")
    print(f"{'-' * 75}")
    print(f"{'Class':<20} {thresh_header}")
    print("-" * 75)
    for cls, _ in improvements[:5]:
        f1_row = " ".join(f"{r['f1']:.2f}" for r in class_sweep[cls])
        print(f"{cls:<20} {f1_row}")

    # -------------------------------------------------------------------------
    # SAVE OUTPUTS
    # -------------------------------------------------------------------------
    output_dir = Path("results/thresholds")
    output_dir.mkdir(exist_ok=True)

    safe_label = model_label.replace(" ", "_").replace("/", "-")
    map_file = output_dir / f"per_class_thresholds_{safe_label}.json"

    with open(map_file, 'w') as f:
        json.dump({
            "model":            model_label,
            "global_baseline":  global_conf,
            # Ready-to-use {class: threshold} map for inference
            "thresholds": {cls: optimal[cls]['threshold'] for cls in class_list},
            "metrics": {
                cls: {
                    "optimal_threshold": optimal[cls]['threshold'],
                    "optimal_f1":        optimal[cls]['f1'],
                    "optimal_precision": optimal[cls]['precision'],
                    "optimal_recall":    optimal[cls]['recall'],
                    "global_f1":         global_f1_by_class[cls],
                    "delta_f1":          round(optimal[cls]['f1'] - global_f1_by_class[cls], 4),
                    "gt_count":          gt_counts.get(cls, 0),
                }
                for cls in class_list
            },
            "full_sweep": {cls: class_sweep[cls] for cls in class_list},
        }, f, indent=2)

    print(f"\n{'=' * 75}")
    print("SUMMARY")
    print(f"{'=' * 75}")
    print(f"  Macro F1 — global {global_conf:.3f}:  {macro_global:.4f}")
    print(f"  Macro F1 — per-class optimal:  {macro_opt:.4f}  ({macro_opt - macro_global:+.4f})")
    print(f"\n  Threshold map saved to: {map_file}")
    print("  Load 'thresholds' key from this file to apply per-class thresholds at inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RF-DETR confidence threshold sweep')
    parser.add_argument('--new-val', action='store_true',
                        help='Use new validation set (test_photos/validation_new_*)')
    parser.add_argument('--val-images', type=str, default=None,
                        help='Custom validation images dir (overrides --new-val path)')
    parser.add_argument('--val-labels', type=str, default=None,
                        help='Custom validation labels dir (overrides --new-val path)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to RF-DETR checkpoint')
    parser.add_argument('--label', type=str, default='RF-DETR',
                        help='Model label for display')
    parser.add_argument('--fine', action='store_true',
                        help='Fine-grained sweep 0.85-0.95 (global sweep only)')
    parser.add_argument('--per-class', action='store_true',
                        help='Run per-class threshold sweep (finds optimal threshold per class)')
    parser.add_argument('--global-conf', type=float, default=0.89,
                        help='Global baseline threshold for per-class comparison (default: 0.89)')
    parser.add_argument('--min-conf', type=float, default=0.500,
                        help='Lower bound of threshold sweep (default: 0.500)')
    parser.add_argument('--max-conf', type=float, default=0.975,
                        help='Upper bound of threshold sweep (default: 0.975)')
    parser.add_argument('--step', type=float, default=0.025,
                        help='Step size between thresholds (default: 0.025)')
    parser.add_argument('--gt-floor-count', type=int, default=20,
                        help='Apply threshold floor for classes with fewer than this many GT instances (default: 20)')
    parser.add_argument('--gt-floor-thresh', type=float, default=0.50,
                        help='Minimum threshold floor for low-GT classes (default: 0.50)')
    parser.add_argument('--classes', type=str, default=None,
                        help='Path to class_names.json for the model (required for reduced class lists like CL3.1)')
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else None

    class_list = None
    if args.classes:
        with open(args.classes, encoding="utf-8") as f:
            class_list = json.load(f)
        print(f"Class list loaded from {args.classes} ({len(class_list)} classes)")

    val_images = Path(args.val_images) if args.val_images else None
    val_labels = Path(args.val_labels) if args.val_labels else None

    if args.per_class:
        run_per_class_threshold_sweep(
            use_new_val=args.new_val,
            weights_path=weights,
            model_label=args.label,
            global_conf=args.global_conf,
            min_conf=args.min_conf,
            max_conf=args.max_conf,
            step=args.step,
            gt_floor_count=args.gt_floor_count,
            gt_floor_thresh=args.gt_floor_thresh,
            class_list=class_list,
            val_images=val_images,
            val_labels=val_labels,
        )
    else:
        run_threshold_sweep(
            use_new_val=args.new_val,
            weights_path=weights,
            model_label=args.label,
            fine=args.fine,
            class_list=class_list,
            val_images=val_images,
            val_labels=val_labels,
        )
