# Image Annotation Tool

A Python-based GUI tool for creating bounding box annotations with automatic object detection using GroundingDINO.

## Features

- 🖼️ **Image Loading**: Support for JPG, JPEG, PNG, BMP, and GIF formats
- 🤖 **Auto-Detection**: First-pass automatic detection using GroundingDINO
- ✏️ **Interactive Editing**: 
  - Draw bounding boxes by clicking and dragging
  - Move boxes by dragging
  - Resize boxes using corner/edge handles
  - Delete boxes with Delete/Backspace keys
- 🏷️ **Class Labeling**: Assign object classes from customizable dropdown
- 💾 **JSON Export**: Export annotations in standardized format
- 📋 **Batch Processing**: Load existing annotations for editing

## Installation

### Basic Installation (Manual Mode Only)

```bash
cd annotation-tool
pip install -r requirements.txt
```

### Full Installation (with GroundingDINO)

For automatic detection, you need to install GroundingDINO:

```bash
# Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install PyTorch (if not already installed)
# Visit https://pytorch.org for platform-specific instructions

# Install other dependencies
pip install -r requirements.txt
```

You'll also need the GroundingDINO model weights in the parent directory's `weights/` folder.

## Usage

### Starting the Tool

```bash
# Start with file picker
python annotation_tool.py

# Or open a specific image
python annotation_tool.py path/to/image.jpg
```

### Workflow

1. **Load Image**: Click "Open Image" or use File → Open Image
2. **Auto-Detect** (Optional): Click "Auto-Detect (DINO)" to run GroundingDINO
3. **Edit Annotations**:
   - **Draw**: Click and drag to create a new bounding box
   - **Select**: Click on an existing box to select it
   - **Move**: Drag a selected box to move it
   - **Resize**: Drag the corner/edge handles of a selected box
   - **Delete**: Select a box and press Delete or Backspace
   - **Change Class**: Select a box and choose a class from the dropdown
4. **Save**: Click "Save JSON" to export annotations

### Keyboard Shortcuts

- **Delete/Backspace**: Delete selected box
- **Escape**: Deselect all boxes

### JSON Format

The tool exports annotations in the following format:

```json
{
  "image": "image-name.jpg",
  "objects": [
    {
      "class": "chair",
      "bbox": [x1, y1, x2, y2]
    },
    {
      "class": "table",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

Where:
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates

## Customization

### Custom Object Classes

Edit `object_list.py` to define your own object classes:

```python
OBJECT_CLASSES = [
    "my_object_1",
    "my_object_2",
    "my_object_3",
]
```

If `object_list.py` is not present, the tool will use default COCO classes.

### Manual Annotation Mode

If GroundingDINO is not available, the tool will automatically fall back to manual annotation mode. You can still:
- Draw bounding boxes manually
- Edit and label objects
- Save annotations to JSON

## Output

The tool saves annotations as JSON files with the same base name as the input image:

```
input: kitchen2.jpg
output: kitchen2.json
```

You can also choose a custom output location when saving.

## Troubleshooting

### "GroundingDINO Not Available"

This is normal if you haven't installed GroundingDINO. The tool will work in manual mode.

To enable auto-detection:
1. Install GroundingDINO: `pip install git+https://github.com/IDEA-Research/GroundingDINO.git`
2. Ensure model weights are in `../weights/` directory
3. Restart the tool

### "Failed to load image"

Make sure:
- Image file exists and is accessible
- Image format is supported (JPG, JPEG, PNG, BMP, GIF)
- Image is not corrupted

### Bounding boxes appear shifted

This shouldn't happen as the tool handles scaling automatically. If it does:
1. Check that you're using the latest version
2. Try reloading the image
3. Report the issue with the image dimensions

## Examples

See the parent directory's `test_photos/` folder for example images and annotations.

## Requirements

- Python 3.8+
- Pillow (for image handling)
- tkinter (usually included with Python)

Optional:
- PyTorch (for GroundingDINO)
- GroundingDINO (for auto-detection)
- OpenCV (for GroundingDINO)

## License

This tool is part of the AI Models Comparison project.

