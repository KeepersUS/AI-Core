"""
STR (Short-Term Rental) Detection Visualization

This module provides visualization utilities for debugging and analyzing
detection results.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image

from str_config import STR_CORE_CLASSES, normalize_label


# Color palette for different classes (BGR format for OpenCV)
CLASS_COLORS = {
    "bed": (255, 100, 100),
    "pillow": (200, 150, 255),
    "couch": (100, 150, 255),
    "chair": (150, 200, 100),
    "table": (100, 255, 200),
    "blanket": (255, 200, 150),
    "television": (50, 50, 200),
    "lamp": (0, 255, 255),
    "mirror": (200, 200, 200),
    "curtain": (255, 150, 100),
    "rug": (150, 100, 50),
    "sink": (200, 200, 0),
    "refrigerator": (0, 150, 255),
    "toilet": (255, 255, 100),
    "shower": (100, 255, 255),
    "towel": (255, 100, 255),
    "toilet paper roll": (200, 100, 200),
    "trashcan": (100, 100, 100),
}

# Status colors for comparison visualization
STATUS_COLORS = {
    "matched": (0, 255, 0),      # Green
    "missing": (0, 0, 255),       # Red
    "extra": (0, 165, 255),       # Orange
    "moved": (255, 255, 0),       # Cyan
}


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """Get the color for a class, generating one if not predefined."""
    normalized = normalize_label(class_name)
    if normalized in CLASS_COLORS:
        return CLASS_COLORS[normalized]
    
    # Generate a deterministic color from the class name
    hash_val = hash(normalized) % (256 ** 3)
    return (hash_val % 256, (hash_val // 256) % 256, (hash_val // 65536) % 256)


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True,
    show_class: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detection bounding boxes on an image.
    
    Args:
        image: Image as numpy array (BGR format)
        detections: List of detection dictionaries
        show_confidence: Whether to show confidence scores
        show_class: Whether to show class labels
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of text labels
        
    Returns:
        Image with detections drawn
    """
    result = image.copy()
    
    for det in detections:
        class_name = det.get('class', 'unknown')
        bbox = det.get('bbox', [0, 0, 0, 0])
        confidence = det.get('confidence', 1.0)
        
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = get_class_color(class_name)
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_thickness)
        
        # Build label text
        label_parts = []
        if show_class:
            label_parts.append(class_name)
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get label size for background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            
            # Draw label background
            label_y = max(y1 - 5, label_h + 5)
            cv2.rectangle(
                result,
                (x1, label_y - label_h - 5),
                (x1 + label_w + 5, label_y + 5),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
    
    return result


def draw_comparison(
    reference_image: np.ndarray,
    postcleaning_image: np.ndarray,
    comparison_result: Dict,
    show_legend: bool = True,
) -> np.ndarray:
    """
    Create a side-by-side comparison visualization.
    
    Args:
        reference_image: Reference image (BGR)
        postcleaning_image: Post-cleaning image (BGR)
        comparison_result: Comparison result dictionary
        show_legend: Whether to show color legend
        
    Returns:
        Combined visualization image
    """
    # Resize images to same height if needed
    h1, w1 = reference_image.shape[:2]
    h2, w2 = postcleaning_image.shape[:2]
    
    target_h = max(h1, h2)
    
    if h1 != target_h:
        scale = target_h / h1
        reference_image = cv2.resize(reference_image, (int(w1 * scale), target_h))
    
    if h2 != target_h:
        scale = target_h / h2
        postcleaning_image = cv2.resize(postcleaning_image, (int(w2 * scale), target_h))
    
    # Draw on reference image
    ref_viz = reference_image.copy()
    
    # Draw matched objects in green
    for obj in comparison_result.get('matched_objects', []):
        if 'matched_bbox' in obj:
            bbox = [int(v) for v in obj['matched_bbox']]
            cv2.rectangle(ref_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         STATUS_COLORS['matched'], 2)
    
    # Draw missing objects in red (on reference)
    for obj in comparison_result.get('missing_objects', []):
        bbox = [int(v) for v in obj['bbox']]
        cv2.rectangle(ref_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     STATUS_COLORS['missing'], 3)
        # Add X mark
        cv2.line(ref_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), STATUS_COLORS['missing'], 2)
        cv2.line(ref_viz, (bbox[2], bbox[1]), (bbox[0], bbox[3]), STATUS_COLORS['missing'], 2)
    
    # Draw on post-cleaning image
    post_viz = postcleaning_image.copy()
    
    # Draw matched objects in green
    for obj in comparison_result.get('matched_objects', []):
        bbox = [int(v) for v in obj['bbox']]
        cv2.rectangle(post_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     STATUS_COLORS['matched'], 2)
    
    # Draw extra objects in orange
    for obj in comparison_result.get('extra_objects', []):
        bbox = [int(v) for v in obj['bbox']]
        cv2.rectangle(post_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     STATUS_COLORS['extra'], 3)
    
    # Draw moved objects in cyan
    for obj in comparison_result.get('moved_objects', []):
        bbox = [int(v) for v in obj['bbox']]
        cv2.rectangle(post_viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     STATUS_COLORS['moved'], 2)
    
    # Add labels
    cv2.putText(ref_viz, "REFERENCE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(post_viz, "POST-CLEANING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Combine side by side
    combined = np.hstack([ref_viz, post_viz])
    
    # Add legend if requested
    if show_legend:
        legend_h = 100
        legend = np.zeros((legend_h, combined.shape[1], 3), dtype=np.uint8)
        legend.fill(40)  # Dark gray background
        
        y_pos = 30
        x_pos = 20
        
        for status, color in STATUS_COLORS.items():
            cv2.rectangle(legend, (x_pos, y_pos - 15), (x_pos + 20, y_pos + 5), color, -1)
            cv2.putText(legend, status.upper(), (x_pos + 30, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            x_pos += 150
        
        # Add summary
        summary = comparison_result.get('summary', {})
        summary_text = (f"Matched: {summary.get('matched', 0)} | "
                       f"Missing: {summary.get('missing', 0)} | "
                       f"Extra: {summary.get('extra', 0)} | "
                       f"Moved: {summary.get('moved', 0)}")
        cv2.putText(legend, summary_text, (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Status indicator
        is_clean = comparison_result.get('is_clean', False)
        status_text = "STATUS: PASS" if is_clean else "STATUS: FAIL"
        status_color = (0, 255, 0) if is_clean else (0, 0, 255)
        cv2.putText(legend, status_text, (combined.shape[1] - 200, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        
        combined = np.vstack([combined, legend])
    
    return combined


def visualize_detections(
    image_path: Union[str, Path],
    detections: List[Dict],
    output_path: Union[str, Path] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Visualize detections on an image.
    
    Args:
        image_path: Path to input image
        detections: List of detection dictionaries
        output_path: Path to save visualization (optional)
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    result = draw_detections(image, detections)
    
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"[Visualizer] Saved to {output_path}")
    
    if show:
        cv2.imshow("Detections", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def visualize_comparison(
    reference_image_path: Union[str, Path],
    postcleaning_image_path: Union[str, Path],
    comparison_result: Dict,
    output_path: Union[str, Path] = None,
    show: bool = False,
) -> np.ndarray:
    """
    Visualize comparison between reference and post-cleaning images.
    
    Args:
        reference_image_path: Path to reference image
        postcleaning_image_path: Path to post-cleaning image
        comparison_result: Comparison result dictionary
        output_path: Path to save visualization (optional)
        show: Whether to display the image
        
    Returns:
        Visualization image
    """
    ref_image = cv2.imread(str(reference_image_path))
    post_image = cv2.imread(str(postcleaning_image_path))
    
    if ref_image is None:
        raise ValueError(f"Could not load reference image: {reference_image_path}")
    if post_image is None:
        raise ValueError(f"Could not load post-cleaning image: {postcleaning_image_path}")
    
    result = draw_comparison(ref_image, post_image, comparison_result)
    
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"[Visualizer] Saved comparison to {output_path}")
    
    if show:
        cv2.imshow("Comparison", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return result


def create_detection_grid(
    image_paths: List[Union[str, Path]],
    predictions: Dict[str, List[Dict]],
    output_path: Union[str, Path] = None,
    grid_cols: int = 4,
    cell_size: Tuple[int, int] = (400, 300),
) -> np.ndarray:
    """
    Create a grid visualization of multiple images with their detections.
    
    Args:
        image_paths: List of image paths
        predictions: Dictionary mapping image paths to detections
        output_path: Path to save grid (optional)
        grid_cols: Number of columns in grid
        cell_size: Size of each cell (width, height)
        
    Returns:
        Grid visualization image
    """
    n_images = len(image_paths)
    n_rows = (n_images + grid_cols - 1) // grid_cols
    
    cell_w, cell_h = cell_size
    grid_w = cell_w * grid_cols
    grid_h = cell_h * n_rows
    
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid.fill(30)  # Dark background
    
    for i, image_path in enumerate(image_paths):
        row = i // grid_cols
        col = i % grid_cols
        
        # Load and resize image
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        
        # Get detections for this image
        detections = predictions.get(str(image_path), [])
        
        # Draw detections
        image = draw_detections(image, detections, font_scale=0.4, line_thickness=1)
        
        # Resize to fit cell
        h, w = image.shape[:2]
        scale = min(cell_w / w, cell_h / h) * 0.9
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image = cv2.resize(image, (new_w, new_h))
        
        # Calculate position to center in cell
        x_offset = col * cell_w + (cell_w - new_w) // 2
        y_offset = row * cell_h + (cell_h - new_h) // 2
        
        # Place in grid
        grid[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = image
        
        # Add filename label
        filename = Path(image_path).stem[:25]
        cv2.putText(grid, filename, (col * cell_w + 5, (row + 1) * cell_h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Add detection count
        count_text = f"{len(detections)} obj"
        cv2.putText(grid, count_text, (col * cell_w + 5, row * cell_h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1, cv2.LINE_AA)
    
    if output_path:
        cv2.imwrite(str(output_path), grid)
        print(f"[Visualizer] Saved grid to {output_path}")
    
    return grid


if __name__ == "__main__":
    print("STR Visualizer module loaded successfully.")
    print("Available functions:")
    print("  - visualize_detections(image_path, detections)")
    print("  - visualize_comparison(ref_path, post_path, comparison_result)")
    print("  - create_detection_grid(image_paths, predictions)")
