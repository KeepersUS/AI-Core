#!/usr/bin/env python3
"""
Generate pseudo-labels using OFFICIAL Grounding DINO with your existing weights.

This uses the official Grounding DINO implementation (from grounding_dino.py)
which loads weights from the weights/ folder. This should produce correctly
placed boxes unlike the MMDetection wrapper.

Usage:
    python scripts/generate_pseudo_labels_official_weights.py \
      --prompt "bed . chair . couch" \
      --threshold 0.5 \
      --max-detections 20
"""

import os
import json
import sys
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import argparse

# Add project root to path to import grounding_dino
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from grounding_dino import GroundingDINODetector
    GROUNDING_DINO_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: Could not import GroundingDINODetector: {e}")
    print(f"   Make sure grounding_dino.py is in {PROJECT_ROOT}")
    print(f"   Also ensure the official Grounding DINO package is installed:")
    print(f"   pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")
    GROUNDING_DINO_AVAILABLE = False
    sys.exit(1)

# Default paths
IMAGE_DIR = Path(__file__).parent.parent / "data" / "unlabeled_images"
OUTPUT_JSON = Path(__file__).parent.parent / "data" / "annotations" / "pseudo_labels_official.json"
WEIGHTS_DIR = PROJECT_ROOT / "weights"


def apply_nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections."""
    if len(detections) <= 1:
        return detections
    
    try:
        import torch
        from torchvision.ops import nms
        
        # Group by class
        detections_by_class = {}
        for idx, det in enumerate(detections):
            class_name = det.get('object', det.get('class', 'unknown'))
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            detections_by_class[class_name].append((idx, det))
        
        keep_indices = set()
        
        # Apply NMS per class
        for class_name, class_dets in detections_by_class.items():
            if len(class_dets) <= 1:
                keep_indices.add(class_dets[0][0])
                continue
            
            boxes = []
            scores = []
            indices = []
            
            for idx, det in class_dets:
                bbox = det['bbox']  # [x1, y1, x2, y2]
                boxes.append(bbox)
                scores.append(det['confidence'])
                indices.append(idx)
            
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)
            
            keep_local = nms(boxes_tensor, scores_tensor, iou_threshold=iou_threshold)
            
            for local_idx in keep_local.tolist():
                keep_indices.add(indices[local_idx])
        
        return [detections[i] for i in sorted(keep_indices)]
    
    except ImportError:
        print("⚠️  Warning: torchvision.ops.nms not available, skipping NMS")
        return detections
    except Exception as e:
        print(f"⚠️  Warning: NMS failed ({e}), using all detections")
        return detections


def main():
    parser = argparse.ArgumentParser(
        description='Generate pseudo-labels using official Grounding DINO with existing weights',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--prompt', type=str, required=True,
                       help='Text prompt with classes separated by " . " (e.g., "bed . chair . couch")')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--max-detections', type=int, default=20,
                       help='Maximum detections per image after NMS (default: 20)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')
    parser.add_argument('--box-threshold', type=float, default=0.3,
                       help='Box threshold for Grounding DINO (default: 0.3)')
    parser.add_argument('--text-threshold', type=float, default=0.3,
                       help='Text threshold for Grounding DINO (default: 0.3)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    parser.add_argument('--test-single', type=str, default=None,
                       help='Test on a single image file (for debugging)')
    
    args = parser.parse_args()
    
    if not GROUNDING_DINO_AVAILABLE:
        sys.exit(1)
    
    output_json = Path(args.output) if args.output else OUTPUT_JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse classes from prompt
    classes = [c.strip() for c in args.prompt.split(' . ')]
    
    print("=" * 70)
    print("OFFICIAL GROUNDING DINO PSEUDO-LABEL GENERATOR")
    print("(Using weights from weights/ folder)")
    print("=" * 70)
    print(f"Prompt: {args.prompt}")
    print(f"Classes: {len(classes)}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Max detections: {args.max_detections}")
    print(f"Box threshold: {args.box_threshold}")
    print(f"Text threshold: {args.text_threshold}")
    print(f"Weights: {WEIGHTS_DIR}")
    print(f"Output: {output_json}")
    print("=" * 70)
    
    # Verify weights exist
    weights_file = WEIGHTS_DIR / "groundingdino_swint_ogc.pth"
    config_file = WEIGHTS_DIR / "GroundingDINO_SwinT_OGC.py"
    
    if not weights_file.exists():
        print(f"❌ Weights file not found: {weights_file}")
        sys.exit(1)
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        sys.exit(1)
    
    print(f"✅ Found weights: {weights_file}")
    print(f"✅ Found config: {config_file}")
    
    if not IMAGE_DIR.exists():
        print(f"❌ Image directory not found: {IMAGE_DIR}")
        sys.exit(1)
    
    # Initialize detector
    print("\n🔧 Initializing Grounding DINO detector...")
    try:
        detector = GroundingDINODetector(device="auto")
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("\n💡 Make sure the official Grounding DINO package is installed:")
        print("   pip install 'git+https://github.com/IDEA-Research/GroundingDINO.git'")
        sys.exit(1)
    
    if detector.model is None:
        print("❌ Failed to load Grounding DINO model")
        print("\n💡 Check:")
        print(f"   1. Weights file exists: {weights_file.exists()}")
        print(f"   2. Config file exists: {config_file.exists()}")
        print(f"   3. Official Grounding DINO package is installed")
        sys.exit(1)
    
    # Override thresholds
    detector.box_threshold = args.box_threshold
    detector.text_threshold = args.text_threshold
    detector.confidence_threshold = args.threshold
    
    print("✅ Grounding DINO initialized successfully!")
    
    # Get image files
    image_files = [f for f in IMAGE_DIR.iterdir() 
                   if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
    
    if args.test_single:
        image_files = [f for f in image_files if args.test_single in f.name]
        if not image_files:
            print(f"❌ Image not found: {args.test_single}")
            sys.exit(1)
        print(f"\n🧪 TEST MODE: Processing only {image_files[0].name}")
    
    print(f"\n📸 Found {len(image_files)} images to process")
    
    # Initialize COCO output
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    classes_dict = {}  # class_name -> category_id
    ann_id = 1
    
    # Statistics
    total_detections = 0
    filtered_detections = 0
    nms_removed = 0
    
    print(f"\n🚀 Processing images...")
    for img_file in tqdm(image_files, desc="Processing"):
        try:
            # Load image to get dimensions
            with Image.open(img_file) as img:
                width, height = img.size
            
            image_id = len(coco_output["images"]) + 1
            
            # Add image info
            coco_output["images"].append({
                "id": image_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            })
            
            # Detect objects using official Grounding DINO directly
            # Use the predict function directly to avoid INDOOR_BUSINESS_CLASSES filtering
            from groundingdino.util.inference import load_image, predict
            
            image_source, image = load_image(str(img_file))
            h, w, _ = image_source.shape
            
            # Create text query from classes
            text_query = ". ".join(classes) + "."
            
            # Run detection
            boxes, confidences, labels = predict(
                model=detector.model,
                image=image,
                caption=text_query,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                device="cpu"
            )
            
            # Convert to detection format
            detections = []
            for box, confidence, label in zip(boxes, confidences, labels):
                if confidence >= args.threshold:
                    cx_norm, cy_norm, w_norm, h_norm = box
                    
                    cx = cx_norm * w
                    cy = cy_norm * h
                    box_w = w_norm * w
                    box_h = h_norm * h
                    
                    x1 = int(cx - box_w / 2)
                    y1 = int(cy - box_h / 2)
                    x2 = int(cx + box_w / 2)
                    y2 = int(cy + box_h / 2)
                    
                    # Clamp to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        class_name = label.strip().lower()
                        detections.append({
                            "object": class_name,
                            "confidence": float(confidence),
                            "bbox": [x1, y1, x2, y2]
                        })
            
            if not detections:
                if args.verbose:
                    print(f"\n   No detections for {img_file.name}")
                continue
            
            total_detections += len(detections)
            
            # Filter by confidence (already done above, but keep for consistency)
            filtered = [det for det in detections if det.get('confidence', 0) >= args.threshold]
            filtered_detections += len(filtered)
            
            # Apply NMS
            num_before_nms = len(filtered)
            filtered = apply_nms(filtered, iou_threshold=0.5)
            num_after_nms = len(filtered)
            nms_removed += (num_before_nms - num_after_nms)
            
            # Limit to max_detections
            if len(filtered) > args.max_detections:
                filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                filtered = filtered[:args.max_detections]
            
            # Convert to COCO format
            for det in filtered:
                class_name = det.get('object', 'unknown')
                bbox = det['bbox']  # [x1, y1, x2, y2]
                confidence = det.get('confidence', 0.0)
                
                # Get or create category ID
                if class_name not in classes_dict:
                    classes_dict[class_name] = len(classes_dict) + 1
                
                category_id = classes_dict[class_name]
                
                # Convert bbox to COCO format [x, y, width, height]
                x1, y1, x2, y2 = bbox
                width_bbox = x2 - x1
                height_bbox = y2 - y1
                
                if width_bbox > 0 and height_bbox > 0:
                    coco_output["annotations"].append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [float(x1), float(y1), float(width_bbox), float(height_bbox)],
                        "area": float(width_bbox * height_bbox),
                        "iscrowd": 0,
                        "score": float(confidence)
                    })
                    ann_id += 1
            
            # Debug output for first image
            if image_id == 1 and args.verbose:
                print(f"\n   First image: {img_file.name}")
                print(f"      Raw detections: {len(detections)}")
                print(f"      After threshold: {num_before_nms}")
                print(f"      After NMS: {num_after_nms}")
                print(f"      Final annotations: {len([a for a in coco_output['annotations'] if a['image_id'] == image_id])}")
                
                # Show class distribution
                class_counts = {}
                for det in filtered:
                    class_name = det.get('object', 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                print(f"      Classes found: {list(class_counts.keys())}")
        
        except Exception as e:
            print(f"\n⚠️  Error processing {img_file.name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue
    
    # Create categories
    for class_name, cat_id in sorted(classes_dict.items(), key=lambda x: x[1]):
        coco_output["categories"].append({
            "id": cat_id,
            "name": class_name,
            "supercategory": "none"
        })
    
    # Save output
    print(f"\n💾 Saving annotations to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(coco_output, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Images processed: {len(coco_output['images'])}")
    print(f"Total detections: {total_detections}")
    print(f"After threshold: {filtered_detections}")
    print(f"NMS removed: {nms_removed} duplicates")
    print(f"Final annotations: {len(coco_output['annotations'])}")
    print(f"Categories: {len(coco_output['categories'])}")
    
    if len(coco_output['images']) > 0:
        avg = len(coco_output['annotations']) / len(coco_output['images'])
        print(f"Average per image: {avg:.1f}")
    
    # Quality check
    if len(coco_output['annotations']) == 0:
        print("\n⚠️  WARNING: No annotations generated!")
        print("   Try lowering the threshold or checking the prompt")
    
    print(f"\n✅ Complete! Output: {output_json}")
    print("=" * 70)


if __name__ == "__main__":
    main()

