"""
Object Classes for Annotation Tool
------------------------------------
Define your custom object classes here.
The annotation tool will use this list for the class dropdown.

If this file is not present, the tool will use default classes
from COCO_CLASSES.py or a basic set of common objects.
"""

# Example: Indoor/Business objects
OBJECT_CLASSES = [
    # Furniture
    "chair",
    "table",
    "desk",
    "couch",
    "bed",
    "cabinet",
    "shelf",
    "bookshelf",
    
    # Technology
    "laptop",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "tv",
    "cell phone",
    "remote",
    
    # Kitchen
    "sink",
    "refrigerator",
    "microwave",
    "oven",
    "toaster",
    "dining table",
    
    # Decor & Plants
    "potted plant",
    "vase",
    "clock",
    "picture",
    "mirror",
    
    # Personal items
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "bowl",
    
    # Books & Office
    "book",
    "scissors",
    
    # Bathroom
    "toilet",
    "hair drier",
    "toothbrush",
]

# Alternative: Simple custom list
# OBJECT_CLASSES = [
#     "object1",
#     "object2",
#     "object3",
# ]

# You can also organize by category if needed
CATEGORIES = {
    "furniture": ["chair", "table", "desk", "couch", "bed"],
    "technology": ["laptop", "monitor", "keyboard", "mouse", "tv"],
    "kitchen": ["sink", "refrigerator", "microwave"],
    "decor": ["potted plant", "vase", "clock"],
}

