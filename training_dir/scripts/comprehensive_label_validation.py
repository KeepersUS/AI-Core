#!/usr/bin/env python3
"""
Comprehensive validation of pseudo-label quality.
Checks for the specific issues we've been fixing:
- Label diversity (not all one class)
- NMS effectiveness (no duplicates)
- Score distribution
- Bbox quality
"""
import json
from pathlib import Path
from collections import Counter, defaultdict
import random
import sys

def validate_labels(json_file: Path):
    """Comprehensive validation of pseudo-labels"""
    
    if not json_file.exists():
        print(f"❌ Error: Label file not found at {json_file}")
        print(f"\nAvailable files in directory:")
        parent_dir = json_file.parent
        if parent_dir.exists():
            for f in sorted(parent_dir.glob("*.json")):
                print(f"   - {f.name}")
        return
    
    print("=" * 70)
    print("COMPREHENSIVE PSEUDO-LABEL QUALITY VALIDATION")
    print("=" * 70)
    print(f"\n📁 Loading labels from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data.get("images", [])}
    annotations = data.get("annotations", [])
    categories = {cat['id']: cat['name'] for cat in data.get("categories", [])}
    
    print(f"✅ Loaded {len(images)} images, {len(annotations)} annotations, {len(categories)} categories\n")
    
    # 1. BASIC STATISTICS
    print("=" * 70)
    print("1. BASIC STATISTICS")
    print("=" * 70)
    print(f"   Total images: {len(images)}")
    print(f"   Total annotations: {len(annotations)}")
    print(f"   Categories: {len(categories)}")
    
    if len(annotations) > 0:
        print(f"   Average annotations per image: {len(annotations) / len(images):.1f}")
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in annotations:
        annotations_by_image[ann['image_id']].append(ann)
    
    ann_counts = [len(anns) for anns in annotations_by_image.values()]
    if ann_counts:
        print(f"   Min annotations per image: {min(ann_counts)}")
        print(f"   Max annotations per image: {max(ann_counts)}")
        images_with_0 = len(images) - len(annotations_by_image)
        print(f"   Images with 0 annotations: {images_with_0} ({images_with_0/len(images)*100:.1f}%)")
    
    # 2. LABEL DIVERSITY (CRITICAL CHECK)
    print("\n" + "=" * 70)
    print("2. LABEL DIVERSITY CHECK (Critical)")
    print("=" * 70)
    
    single_class_images = 0
    multi_class_images = 0
    class_diversity_stats = []
    
    for img_id, anns in annotations_by_image.items():
        unique_classes = set(ann['category_id'] for ann in anns)
        num_unique_classes = len(unique_classes)
        class_diversity_stats.append(num_unique_classes)
        
        if num_unique_classes == 1:
            single_class_images += 1
        else:
            multi_class_images += 1
    
    if class_diversity_stats:
        print(f"   Images with SINGLE class: {single_class_images} ({single_class_images/len(annotations_by_image)*100:.1f}%)")
        print(f"   Images with MULTIPLE classes: {multi_class_images} ({multi_class_images/len(annotations_by_image)*100:.1f}%)")
        print(f"   Average unique classes per image: {sum(class_diversity_stats)/len(class_diversity_stats):.2f}")
        
        if single_class_images > len(annotations_by_image) * 0.2:
            print(f"\n   ⚠️  WARNING: {single_class_images/len(annotations_by_image)*100:.1f}% of images have only ONE class!")
            print(f"      This suggests poor label diversity.")
        
        # Show examples of single-class images
        if single_class_images > 0:
            single_class_examples = []
            for img_id, anns in list(annotations_by_image.items())[:20]:
                unique_classes = set(ann['category_id'] for ann in anns)
                if len(unique_classes) == 1:
                    class_id = list(unique_classes)[0]
                    class_name = categories.get(class_id, f"Unknown-{class_id}")
                    img_info = images.get(img_id, {})
                    single_class_examples.append((img_info.get('file_name', f'image_{img_id}'), class_name, len(anns)))
            
            if single_class_examples:
                print(f"\n   📋 Examples of single-class images (first 5):")
                for img_name, class_name, num_anns in single_class_examples[:5]:
                    print(f"      - {img_name}: {num_anns}x '{class_name}'")
    
    # 3. CLASS DISTRIBUTION
    print("\n" + "=" * 70)
    print("3. CLASS DISTRIBUTION")
    print("=" * 70)
    
    class_counts = Counter([ann['category_id'] for ann in annotations])
    print(f"   Top 15 Most Detected Classes:")
    for i, (cat_id, count) in enumerate(class_counts.most_common(15), 1):
        class_name = categories.get(cat_id, f"Unknown-{cat_id}")
        percentage = (count / len(annotations)) * 100 if annotations else 0
        print(f"      {i:2d}. {class_name:25s}: {count:5d} ({percentage:5.1f}%)")
    
    # Check for class imbalance
    if len(class_counts) > 0:
        most_common = class_counts.most_common(1)[0][1]
        percentage = (most_common / len(annotations)) * 100 if annotations else 0
        if percentage > 30:
            top_class = categories.get(class_counts.most_common(1)[0][0], "Unknown")
            print(f"\n   ⚠️  WARNING: Top class '{top_class}' is {percentage:.1f}% of all detections")
            print(f"      This suggests class imbalance or model bias.")
    
    # 4. CONFIDENCE SCORES
    print("\n" + "=" * 70)
    print("4. CONFIDENCE SCORE ANALYSIS")
    print("=" * 70)
    
    scores = [ann.get('score') for ann in annotations if 'score' in ann]
    
    if scores:
        valid_scores = [s for s in scores if s is not None]
        print(f"   Annotations with scores: {len(valid_scores)}/{len(annotations)} ({len(valid_scores)/len(annotations)*100:.1f}%)")
        
        if valid_scores:
            print(f"   Score range: Min={min(valid_scores):.4f}, Max={max(valid_scores):.4f}")
            print(f"   Score mean: {sum(valid_scores)/len(valid_scores):.4f}")
            print(f"   Score median: {sorted(valid_scores)[len(valid_scores)//2]:.4f}")
            
            # Check if all scores are the same (suspicious)
            if len(set(valid_scores)) == 1:
                print(f"\n   ⚠️  WARNING: All scores are identical ({valid_scores[0]:.4f})!")
                print(f"      This suggests scores are not real confidence values.")
            elif max(valid_scores) - min(valid_scores) < 0.01:
                print(f"\n   ⚠️  WARNING: Scores have very little variation!")
    else:
        print(f"   ⚠️  No confidence scores found in annotations")
        print(f"      Scores are needed for quality filtering.")
    
    # 5. BBOX QUALITY
    print("\n" + "=" * 70)
    print("5. BBOX QUALITY CHECKS")
    print("=" * 70)
    
    bbox_stats = []
    tiny_boxes = 0
    huge_boxes = 0
    
    for ann in annotations:
        img_id = ann['image_id']
        img_info = images.get(img_id)
        if img_info:
            bbox = ann['bbox']  # [x, y, width, height]
            area = ann.get('area', bbox[2] * bbox[3])
            img_area = img_info['width'] * img_info['height']
            relative_area = area / img_area if img_area > 0 else 0
            bbox_stats.append(relative_area)
            
            if relative_area < 0.01:  # < 1% of image
                tiny_boxes += 1
            elif relative_area > 0.5:  # > 50% of image
                huge_boxes += 1
    
    if bbox_stats:
        print(f"   Relative bbox area statistics:")
        print(f"      Min: {min(bbox_stats)*100:.3f}%")
        print(f"      Max: {max(bbox_stats)*100:.3f}%")
        print(f"      Mean: {sum(bbox_stats)/len(bbox_stats)*100:.3f}%")
        print(f"      Median: {sorted(bbox_stats)[len(bbox_stats)//2]*100:.3f}%")
        print(f"\n   Tiny boxes (<1% of image): {tiny_boxes} ({tiny_boxes/len(bbox_stats)*100:.1f}%)")
        print(f"   Huge boxes (>50% of image): {huge_boxes} ({huge_boxes/len(bbox_stats)*100:.1f}%)")
        
        if tiny_boxes > len(bbox_stats) * 0.3:
            print(f"\n   ⚠️  WARNING: {tiny_boxes/len(bbox_stats)*100:.1f}% of boxes are very small (<1%)")
            print(f"      This might indicate false positives or noise.")
    
    # 6. DUPLICATE DETECTION CHECK (NMS effectiveness)
    print("\n" + "=" * 70)
    print("6. DUPLICATE DETECTION CHECK (NMS Effectiveness)")
    print("=" * 70)
    
    def calculate_iou(bbox1, bbox2):
        """Calculate IoU between two bboxes in [x, y, width, height] format"""
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        union = (w1 * h1) + (w2 * h2) - intersection
        return intersection / union if union > 0 else 0.0
    
    duplicate_pairs = 0
    high_iou_pairs = 0
    samples_checked = 0
    max_samples = 100  # Check first 100 images
    
    for img_id, anns in list(annotations_by_image.items())[:max_samples]:
        samples_checked += 1
        # Group by class
        anns_by_class = defaultdict(list)
        for ann in anns:
            anns_by_class[ann['category_id']].append(ann)
        
        # Check for duplicates within each class
        for class_id, class_anns in anns_by_class.items():
            if len(class_anns) > 1:
                for i, ann1 in enumerate(class_anns):
                    for ann2 in class_anns[i+1:]:
                        iou = calculate_iou(ann1['bbox'], ann2['bbox'])
                        if iou > 0.3:  # Overlapping boxes
                            duplicate_pairs += 1
                        if iou > 0.5:  # High overlap (should be removed by NMS)
                            high_iou_pairs += 1
    
    print(f"   Checked {samples_checked} images for duplicate detections")
    print(f"   Overlapping boxes (IoU > 0.3): {duplicate_pairs} pairs")
    print(f"   High overlap boxes (IoU > 0.5): {high_iou_pairs} pairs")
    
    if high_iou_pairs > 0:
        print(f"\n   ⚠️  WARNING: {high_iou_pairs} high-overlap pairs found (IoU > 0.5)")
        print(f"      NMS should have removed these. NMS may not be working correctly.")
    elif duplicate_pairs == 0:
        print(f"\n   ✅ Good: No duplicate detections found in sample")
    
    # 7. SAMPLE ANNOTATIONS
    print("\n" + "=" * 70)
    print("7. SAMPLE ANNOTATIONS")
    print("=" * 70)
    
    if annotations:
        sampled = random.sample(annotations, min(5, len(annotations)))
        for i, ann in enumerate(sampled, 1):
            img_id = ann['image_id']
            img_info = images.get(img_id, {})
            class_name = categories.get(ann['category_id'], f"Unknown-{ann['category_id']}")
            bbox = ann['bbox']
            score = ann.get('score', 'N/A')
            
            print(f"\n   Sample {i}:")
            print(f"      Image: {img_info.get('file_name', f'image_{img_id}')}")
            print(f"      Class: {class_name}")
            print(f"      BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            print(f"      Score: {score}")
    
    # FINAL ASSESSMENT
    print("\n" + "=" * 70)
    print("QUALITY ASSESSMENT")
    print("=" * 70)
    
    issues = []
    warnings = []
    
    if single_class_images > len(annotations_by_image) * 0.2:
        issues.append(f"High percentage of single-class images ({single_class_images/len(annotations_by_image)*100:.1f}%)")
    
    if high_iou_pairs > 0:
        issues.append(f"NMS may not be working ({high_iou_pairs} high-overlap pairs found)")
    
    if not scores or len(valid_scores) == 0:
        warnings.append("No confidence scores stored")
    
    if len(class_counts) > 0:
        top_class_pct = (class_counts.most_common(1)[0][1] / len(annotations)) * 100
        if top_class_pct > 30:
            warnings.append(f"Class imbalance: top class is {top_class_pct:.1f}% of detections")
    
    if tiny_boxes > len(bbox_stats) * 0.3:
        warnings.append(f"Many tiny boxes ({tiny_boxes/len(bbox_stats)*100:.1f}%)")
    
    if issues:
        print("\n   ❌ CRITICAL ISSUES:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("\n   ✅ No critical issues found")
    
    if warnings:
        print("\n   ⚠️  WARNINGS:")
        for warning in warnings:
            print(f"      - {warning}")
    else:
        print("\n   ✅ No warnings")
    
    print("\n" + "=" * 70)
    print("💡 Next Steps:")
    print("=" * 70)
    print("   1. Visualize annotations on actual images to verify box placement")
    print("   2. Manually inspect samples to check label accuracy")
    print("   3. If issues found, review prompt and model configuration")
    print("   4. Consider filtering by confidence scores if available")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate pseudo-label quality")
    parser.add_argument("--labels", type=str, default=None,
                       help="Path to label JSON file (auto-detects if not provided)")
    args = parser.parse_args()
    
    if args.labels:
        label_file = Path(args.labels)
    else:
        # Try to find label files automatically
        script_dir = Path(__file__).parent.parent
        candidates = [
            script_dir / "mmdetection/data/annotations/pseudo_labels_full.json",
            script_dir / "data/annotations/pseudo_labels_full.json",
            script_dir / "data/annotations/pseudo_labels_v1.json",
        ]
        
        found = False
        for candidate in candidates:
            if candidate.exists():
                label_file = candidate
                found = True
                break
        
        if not found:
            print("❌ Error: Could not find label file automatically.")
            print("\nPlease specify with --labels <path> or place file in one of:")
            for c in candidates:
                print(f"   - {c}")
            sys.exit(1)
    
    validate_labels(label_file)

