# Image Annotation Tool

A Python-based GUI tool for creating bounding box annotations on short-term rental property images. Supports both manual annotation and automatic detection via local GroundingDINO model or remote API.

---

## Quick Start

### Windows
```bash
# Double-click to run
run_annotation_tool.bat
```

### Mac/Linux
```bash
cd annotation-tool
pip install Pillow
python3 annotation_tool.py
```

### With a specific image
```bash
python annotation_tool.py ../test_photos/images/kitchen2.jpg
```

---

## Features

### Annotation
- **Draw bounding boxes** - Click and drag to create
- **Move boxes** - Drag selected boxes to reposition
- **Resize boxes** - Drag corner/edge handles
- **Delete boxes** - Delete or Backspace key
- **Class labeling** - Searchable dropdown with 100+ rental property objects

### Auto-Detection
- **Local GroundingDINO** - Run detection on your machine (requires GPU)
- **API Detection** - Send images to remote detection API
- **Batch Processing** - Process entire directories via API

### Navigation
- **Zoom** - Mouse wheel or slider (10% - 500%)
- **Pan** - Right-click and drag
- **Reset View** - One-click reset zoom

### Evaluation
- **Test with Ground Truth** - Compare detections against saved annotations
- **View metrics** - mAP, F1 score, confusion matrix

---

## Installation

### Basic (Manual Annotation Only)
```bash
cd annotation-tool
pip install Pillow
```

### Full (with Local GroundingDINO)
```bash
# Install PyTorch (visit pytorch.org for your platform)
pip install torch torchvision

# Install GroundingDINO
pip install git+https://github.com/IDEA-Research/GroundingDINO.git

# Install other dependencies
pip install -r requirements.txt
```

> **Note**: Local GroundingDINO requires model weights in `../weights/groundingdino_swint_ogc.pth`

### API Mode (Recommended)
No additional installation needed - just configure the API endpoint in Settings.

---

## Usage

### Basic Workflow

1. **Open Image** - Click "Open Image" or File → Open Image
2. **Auto-Detect** (Optional):
   - Click **"Auto-Detect (Local)"** for local GroundingDINO
   - Click **"Auto-Detect (API)"** for remote API detection
3. **Edit Annotations**:
   - Draw new boxes by clicking and dragging
   - Click to select, drag to move
   - Drag handles to resize
   - Press Delete to remove
4. **Label Objects** - Select box, then pick class from dropdown
5. **Save** - Click "Save JSON"

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Delete / Backspace | Delete selected box |
| Escape | Deselect all |

### Mouse Controls

| Action | Effect |
|--------|--------|
| Left-click + drag (empty) | Draw new box |
| Left-click (on box) | Select box |
| Left-drag (selected box) | Move box |
| Left-drag (handle) | Resize box |
| Right-click + drag | Pan canvas |
| Mouse wheel | Zoom in/out |

---

## API Integration

The tool connects to the Grounding DINO API for remote detection.

### Configure API Endpoint

1. Go to **Settings → Configure API Endpoint**
2. Enter API URL (e.g., `https://ai-core-xxx.run.app`)
3. Click **Save**

Default: `https://ai-core-787266927042.us-central1.run.app`

### Test Connection

**Settings → Test API Connection** - Checks `/health` endpoint

### API Detection

Click **"Auto-Detect (API)"** to send the current image to the `/detect` endpoint.

### Batch Processing

Process multiple images at once:

1. Click **"Batch Process Dir"** or Edit → Batch Process Directory (API)
2. Select a folder containing images
3. The tool will:
   - Find all images without existing JSON annotations
   - Send each to the API
   - Save detection results as JSON files

---

## Model Evaluation

Test the API's detection accuracy against your ground truth annotations:

1. Load an image that has a saved `.json` annotation file
2. Go to **Evaluate → Test Photo with Ground Truth**
3. View results:
   - Mean Average Precision (mAP)
   - Mean F1 Score
   - Mean Accuracy
   - Confusion matrix showing correct/incorrect predictions

---

## JSON Format

### Input/Output
```json
{
  "image": "kitchen2.jpg",
  "objects": [
    {
      "class": "refrigerator",
      "bbox": [50, 100, 200, 400]
    },
    {
      "class": "sink",
      "bbox": [300, 150, 450, 350]
    }
  ]
}
```

### Bounding Box Format
`[x1, y1, x2, y2]` - Top-left and bottom-right corners in pixels

---

## Customizing Object Classes

Edit `object_list.py` to define custom classes:

```python
OBJECT_CLASSES = [
    "bed",
    "pillow",
    "refrigerator",
    # Add your classes...
]
```

The tool includes 100+ pre-defined classes for short-term rentals:

| Category | Examples |
|----------|----------|
| Bedroom | bed, pillow, comforter, nightstand, dresser |
| Bathroom | toilet, shower curtain, towel, bath mat |
| Kitchen | refrigerator, oven, microwave, coffee maker |
| Living Area | couch, sofa, coffee table, television |
| Damage | stain, chipping damage, ripping damage, grime |

---

## Menu Reference

### File
- **Open Image** - Load image file
- **Save Annotations** - Export to JSON
- **Exit** - Close application

### Edit
- **Run GroundingDINO (Local)** - Local model detection
- **Run API Detection** - Remote API detection
- **Batch Process Directory (API)** - Process folder
- **Clear All Boxes** - Remove all annotations
- **Delete Selected** - Remove selected box

### Settings
- **Configure API Endpoint** - Set API URL
- **Test API Connection** - Health check

### View
- **Zoom In/Out** - Adjust zoom level
- **Reset Zoom** - Return to 100%
- **Resize Image to 640x480** - Permanently resize image

### Evaluate
- **Test Photo with Ground Truth** - Compare vs saved annotations

---

## Troubleshooting

### "GroundingDINO Not Available"
Normal if not installed. Use **API Detection** instead, or install:
```bash
pip install git+https://github.com/IDEA-Research/GroundingDINO.git
```

### "requests library not found"
Install for API features:
```bash
pip install requests
```

### API Connection Failed
- Check API URL in Settings → Configure API Endpoint
- Test connection with Settings → Test API Connection
- Verify network connectivity

### Image Won't Load
- Supported formats: JPG, JPEG, PNG, BMP, GIF
- Check file isn't corrupted
- Verify file permissions

### Boxes Appear Shifted
The tool handles scaling automatically. Try:
1. Reload the image
2. Reset zoom
3. Check image dimensions match annotation file

---

## File Structure

```
annotation-tool/
├── annotation_tool.py     # Main application
├── object_list.py         # Custom object classes
├── requirements.txt       # Python dependencies
├── run_annotation_tool.bat    # Windows launcher
├── run_annotation_tool.sh     # Unix launcher
├── README.md              # This file
└── QUICKSTART.md          # Quick start guide
```

---

## Requirements

**Required:**
- Python 3.8+
- Pillow (image handling)
- tkinter (usually included with Python)

**Optional:**
- requests (for API features)
- PyTorch + torchvision (for local GroundingDINO)
- OpenCV (for local GroundingDINO)
- GroundingDINO (for local auto-detection)

---

## Tips

1. **Use API Detection** - Faster setup, no GPU required
2. **Batch process first** - Generate initial annotations for all images
3. **Then refine manually** - Fix any detection errors
4. **Search classes** - Type in the search box to filter the class list
5. **Zoom for precision** - Use mouse wheel to zoom in for accurate box placement
6. **Auto-load annotations** - Opening an image automatically loads its JSON if it exists
