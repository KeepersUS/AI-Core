# Quick Start Guide

## Getting Started in 3 Steps

### 1. Install Dependencies

```bash
cd annotation-tool
pip install Pillow
```

That's it for basic usage! The tool will work in manual annotation mode.

### 2. Run the Tool

**Windows:**
- Double-click `run_annotation_tool.bat`

**Mac/Linux:**
```bash
python3 annotation_tool.py
```

**Or with a specific image:**
```bash
python annotation_tool.py ../test_photos/images/kitchen2.jpg
```

### 3. Start Annotating!

1. Click **"Open Image"** to load an image
2. **Draw boxes**: Click and drag on the image
3. **Label**: Select a class from the dropdown on the right
4. **Save**: Click **"Save JSON"** when done

## Optional: Enable Auto-Detection

To use GroundingDINO for automatic first-pass detection:

```bash
# Install PyTorch first (visit pytorch.org for your system)
pip install torch torchvision

# Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install other dependencies
pip install opencv-python numpy
```

Then in the tool, click **"Auto-Detect (DINO)"** after loading an image.

## Controls

### Mouse
- **Click & Drag** on image → Draw new box
- **Click** on box → Select box
- **Drag** selected box → Move box
- **Drag** corner/edge handles → Resize box

### Keyboard
- **Delete** or **Backspace** → Delete selected box
- **Escape** → Deselect all

## Output Format

The tool saves JSON files like this:

```json
{
  "image": "kitchen2.jpg",
  "objects": [
    {
      "class": "chair",
      "bbox": [399, 281, 528, 475]
    },
    {
      "class": "sink",
      "bbox": [49, 237, 244, 384]
    }
  ]
}
```

## Customizing Object Classes

Edit `object_list.py` to add your own classes:

```python
OBJECT_CLASSES = [
    "my_custom_class_1",
    "my_custom_class_2",
    # ... more classes
]
```

## Tips

- **Load existing annotations**: The tool automatically loads `image_name.json` if it exists
- **Edit existing labels**: Just open an image that has a JSON file next to it
- **Keyboard shortcuts**: Use Delete to quickly remove bad boxes
- **Auto-detect first**: Run GroundingDINO first, then refine the results manually

## Troubleshooting

**Problem**: Tool won't start
- **Solution**: Make sure Python 3.8+ is installed and Pillow is installed

**Problem**: Can't see Auto-Detect button working
- **Solution**: GroundingDINO is optional. Install it for auto-detection (see above)

**Problem**: Boxes are in wrong position
- **Solution**: This shouldn't happen - coordinates are automatically scaled. Try reloading the image.

## Examples

Try annotating the test images:

```bash
python annotation_tool.py ../test_photos/images/bedroom1.jpeg
python annotation_tool.py ../test_photos/images/kitchen1.jpg
```

## Next Steps

1. Annotate your images
2. Save the JSON files
3. Use them for training or evaluation!

Enjoy annotating! 🎨

