"""
STR (Short-Term Rental) Object Detection Configuration

Class definitions, label normalization, and evaluation settings shared
across the RF-DETR training and evaluation pipeline.
"""

import json
from pathlib import Path
from typing import List, Dict

# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# Class list loaded from class_names.json (place alongside this file in AI-Core).
# Authoritative source: datasets/CL4/martin_cl4/class_names.json
_CLASS_NAMES_PATH = Path(__file__).parent / "class_names.json"
with open(_CLASS_NAMES_PATH) as _f:
    STR_CORE_CLASSES: List[str] = json.load(_f)

# =============================================================================
# ROOM IMPOSSIBLE CLASSES - Objects that physically cannot be in certain rooms
# =============================================================================
# Instead of "expected" classes (which could miss portable items moved by tenants),
# we define what's IMPOSSIBLE to find in each room type. This prevents false
# negatives for portable objects like blankets in living rooms.

ROOM_IMPOSSIBLE_CLASSES: Dict[str, List[str]] = {
    "bathroom": [
        "bed", "couch", "refrigerator", "oven/stove", "television",
    ],
    "bedroom": [
        "refrigerator", "oven/stove", "toilet", "shower",
    ],
    "living": [
        "refrigerator", "oven/stove", "toilet", "shower", "toilet paper roll",
    ],
    "kitchen": [
        "bed", "toilet", "shower", "toilet paper roll",
    ],
    "outdoor": [
        "bed", "toilet", "shower", "toilet paper roll", "oven/stove",
    ],
}

# Portable classes that could reasonably be found in any indoor room
PORTABLE_CLASSES: List[str] = [
    "pillow", "blanket", "towel", "lamp", "chair", "trashcan", "fan", "rug", "soap",
]

# Fixed/installed classes that don't move between rooms
FIXED_CLASSES: List[str] = [
    "bed", "couch", "refrigerator", "oven/stove", "toilet", "shower", "sink",
    "mirror", "television", "curtain", "rug",
    "dresser",
]

# Label normalization mapping - maps various label formats to standard class names
LABEL_NORMALIZATION: Dict[str, str] = {
    # Trashcan variants
    "trash can": "trashcan",
    "trash": "trashcan",
    "garbage can": "trashcan",
    "garbage bin": "trashcan",
    "waste basket": "trashcan",
    "bin": "trashcan",
    "waste bin": "trashcan",
    "dustbin": "trashcan",
    
    # Television variants
    "tv": "television",
    "TV": "television",
    "flat screen tv": "television",
    "flatscreen tv": "television",
    "monitor": "television",
    "screen": "television",
    
    # Towel variants (critical - was 0% recall)
    "bath towel": "towel",
    "hand towel": "towel",
    "towel hanging": "towel",
    "towel rack": "towel",
    "towels": "towel",
    
    # Shower variants (critical - was 0% recall)
    "shower head": "shower",
    "shower stall": "shower",
    "bathtub": "shower",
    "bath": "shower",
    "tub": "shower",
    "bathroom shower": "shower",
    
    # Blanket variants (was 22% F1)
    "throw blanket": "blanket",
    "comforter": "blanket",
    "duvet": "blanket",
    "bedding": "blanket",
    "bed cover": "blanket",

    #basket variants
    "basket": "basket",
    "laundry basket": "basket",
    "laundry bag": "basket",
    "laundry bin": "basket",
    "laundry box": "basket",
    "laundry container": "basket",
    "laundry tub": "basket",
    "laundry basket": "basket",
    
    # Lamp variants (was 40% recall)
    "light fixture": "lamp",
    "table lamp": "lamp",
    "floor lamp": "lamp",
    "light": "lamp",
    "lighting": "lamp",
    "ceiling light": "lamp",
    "chandelier": "lamp",
    
    # Couch variants
    "sofa": "couch",
    "loveseat": "couch",
    "settee": "couch",
    
    # Toilet paper variants
    "toilet paper": "toilet paper roll",
    "tp roll": "toilet paper roll",
    "toilet roll": "toilet paper roll",
    
    # Refrigerator variants
    "fridge": "refrigerator",
    
    # Table variants
    "end table": "table",
    "coffee table": "table",
    "dining table": "table",
    "desk": "table",
    "counter": "table",
    "countertop": "table",
    
    # Chair variants
    "armchair": "chair",
    "stool": "chair",
    "seat": "chair",
    
    # Curtain variants
    "drape": "curtain",
    "drapes": "curtain",
    "blinds": "curtain",
    "window covering": "curtain",
    
    # Rug variants
    "carpet": "rug",
    "mat": "rug",
    "floor mat": "rug",
    "bath mat": "rug",
    
    # Mirror variants
    "vanity mirror": "mirror",
    "bathroom mirror": "mirror",
    "wall mirror": "mirror",
    
    # Fan variants
    "ceiling fan": "fan",
    "floor fan": "fan",
    "standing fan": "fan",
    "box fan": "fan",
    "exhaust fan": "fan",
    
    # Stove/oven variants (merged into oven/stove in CL4)
    "stove": "oven/stove",
    "oven": "oven/stove",
    "range": "oven/stove",
    "cooktop": "oven/stove",
    "stovetop": "oven/stove",
    "gas stove": "oven/stove",
    "electric stove": "oven/stove",

    #washer and dryer variants
    "washer": "washer/dryer",
    "dryer": "washer/dryer",
    "washing machine": "washer/dryer",
    "laundry machine": "washer/dryer",
    "laundry dryer": "washer/dryer",
    "clothes dryer": "washer/dryer",
    "clothes washer": "washer/dryer",
    "washer dryer": "washer/dryer",
    "washer/dryer": "washer/dryer",

    #plant variants
    "plant": "plant",
    "potted plant": "plant",
    "indoor plant": "plant",
    "outdoor plant": "plant",
    "indoor garden": "plant",
    "outdoor garden": "plant",

    # Decor variants (replaces "picture" from CL1)
    "picture": "decor",
    "art": "decor",
    "wall art": "decor",
    "wall decor": "decor",
    "artwork": "decor",
    "painting": "decor",
    "book": "decor",
    "books": "decor",
    "framed picture": "decor",
    "photo": "decor",

    # Soap variants
    "dish soap": "soap",
    "hand soap": "soap",
    "shampoo": "soap",
    "conditioner": "soap",
    "body wash": "soap",
    "cleaning liquid": "soap",
    "liquid soap": "soap",
    "lotion": "soap",

    # Dresser variants
    "sideboard": "dresser",
    "chest of drawers": "dresser",
    "chest-of-drawers": "dresser",
    "bureau": "dresser",
    "wardrobe": "dresser",

}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# IoU threshold for matching detected objects to ground truth
IOU_THRESHOLD: float = 0.5

def normalize_label(label: str) -> str:
    """
    Normalize a label to its canonical class name.
    
    Args:
        label: Raw label string from detection or annotation
        
    Returns:
        Normalized label string matching STR_CORE_CLASSES
    """
    # Convert to lowercase and strip whitespace
    label_lower = label.lower().strip()

    # Check normalization map
    if label_lower in LABEL_NORMALIZATION:
        return LABEL_NORMALIZATION[label_lower]
    
    # Check if it's already a valid class name
    if label_lower in [c.lower() for c in STR_CORE_CLASSES]:
        # Return the properly cased version
        for c in STR_CORE_CLASSES:
            if c.lower() == label_lower:
                return c
    
    # Return as-is if no mapping found (will be flagged as unknown)
    return label_lower


def get_class_index(class_name: str) -> int:
    """
    Get the index of a class in STR_CORE_CLASSES.
    
    Args:
        class_name: Name of the class
        
    Returns:
        Index in STR_CORE_CLASSES, or -1 if not found
    """
    normalized = normalize_label(class_name)
    try:
        return STR_CORE_CLASSES.index(normalized)
    except ValueError:
        return -1


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device() -> str:
    """
    Get the best available device for inference.
    
    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

