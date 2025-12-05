# Grounding DINO Object Detection API

A FastAPI-based object detection service using [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) for short-term rental property inspection. Detects and evaluates indoor objects (furniture, appliances, amenities) against ground truth annotations.

> ⚠️ **Deployment Notice**: Any push/merge to `main` branch triggers automatic deployment to GCP Cloud Run. Review changes carefully before merging.

---

## Quick Start

### Local Development

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model weights (if not present)
# Place groundingdino_swint_ogc.pth in weights/ folder
# Download from: https://github.com/IDEA-Research/GroundingDINO/releases

# 4. Run the API in local development mode
python dinoAPI.py --local
```

The API will be available at `http://localhost:8080`

#### Local Mode (`--local` or `-l`)

Enables development-friendly settings:
- **Hot reload**: Auto-restarts server when code changes
- **Localhost only**: Binds to `127.0.0.1` (not exposed to network)

```bash
# Local development (recommended)
python dinoAPI.py --local

# Production mode (no reload, binds to 0.0.0.0)
python dinoAPI.py
```

### Docker

```bash
# Build image
docker build -t grounding-dino-api .

# Run with GPU support
docker run --gpus all -p 8080:8080 grounding-dino-api

# Run CPU-only
docker run -p 8080:8080 grounding-dino-api
```

---

## API Endpoints

### Health Check
```
GET /health
```
Returns API status and whether the model is loaded.

### Model Warmup
```
POST /warmup
```
Pre-loads the Grounding DINO model into memory. Call once at startup to avoid cold-start delays.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_gpu` | bool | `true` | Use GPU acceleration if available |

### Object Detection
```
POST /detect
```
Detects objects in an image. Returns bounding boxes and confidence scores.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | ✅ | Image file (JPEG, PNG) |
| `classes` | string | ❌ | Comma-separated list of classes to detect |
| `use_gpu` | bool | ❌ | Use GPU (default: true) |
| `box_threshold` | float | ❌ | Box confidence threshold |
| `text_threshold` | float | ❌ | Text confidence threshold |
| `confidence_threshold` | float | ❌ | Minimum confidence for results |

**Response:**
```json
{
  "status": "ok",
  "image": "bedroom1.jpeg",
  "objects": [
    {"class": "bed", "bbox": [100, 50, 400, 300], "confidence": 0.92},
    {"class": "pillow", "bbox": [150, 80, 250, 150], "confidence": 0.87}
  ],
  "num_detections": 2,
  "execution_time_seconds": 1.23
}
```

### Run Evaluation
```
POST /run
```
Runs detection and compares results against ground truth annotations. Returns confusion matrix metrics.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | file | ✅ | Image file to analyze |
| `reference_json` | file | ✅ | Ground truth JSON annotations |
| `use_gpu` | bool | ❌ | Use GPU (default: true) |
| `create_overlay` | bool | ❌ | Generate visualization overlay (default: true) |

**Reference JSON Format:**
```json
{
  "objects": [
    {"class": "bed", "bbox": [100, 50, 400, 300]},
    {"class": "pillow", "bbox": [150, 80, 250, 150]}
  ]
}
```

**Response:**
```json
{
  "status": "ok",
  "execution_time_seconds": 2.45,
  "comparison_results": {
    "mean_average_precision": 0.85,
    "mean_f1_score": 0.82,
    "unmatched_objects": 1,
    "missing_objects": 0,
    "confusion_pairs": [...]
  },
  "resource_usage_avg": {
    "cpu_percent_avg": 45.2,
    "gpu_utilization_percent_avg": 78.5
  }
}
```

---

## Project Structure

```
├── dinoAPI.py              # FastAPI application & endpoints
├── grounding_dino.py       # Grounding DINO detector implementation
├── model_tests.py          # Confusion matrix & metrics calculation
├── COCO_CLASSES.py         # Indoor object class definitions
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker build configuration
├── weights/                # Model weights & config files
│   ├── groundingdino_swint_ogc.pth
│   └── GroundingDINO_SwinT_OGC.py
├── test_photos/            # Test images and ground truth labels
│   ├── images/
│   └── labels/
├── postman/                # Postman collection for API testing
└── outputs/                # Generated visualizations & results
```

---

## Supported Object Classes

The model detects 117 indoor object classes relevant to short-term rentals, including:

| Category | Examples |
|----------|----------|
| **Bedroom** | bed, pillow, comforter, nightstand, dresser, lamp |
| **Bathroom** | toilet, shower curtain, towel, bath mat, hand soap |
| **Kitchen** | refrigerator, oven, microwave, coffee maker, toaster |
| **Living Area** | couch, sofa, coffee table, television, rug |
| **Laundry** | washer, dryer, laundry detergent, ironing board |
| **Damage/Issues** | stain, chipping damage, ripping damage, grime |

Full list available in `COCO_CLASSES.py`

---

## Configuration

### Detection Thresholds

Default values in `grounding_dino.py`:

| Threshold | Default | Description |
|-----------|---------|-------------|
| `box_threshold` | 0.5 | Minimum confidence for bounding box proposals |
| `text_threshold` | 0.5 | Minimum confidence for text-image matching |
| `confidence_threshold` | 0.5 | Minimum confidence for final detections |

### IoU Settings

In `COCO_CLASSES.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `IOU_THRESHOLD` | 0.25 | Min IoU to match detection with ground truth |
| `AREA_THRESHOLD` | 0.9 | Max area difference ratio for matching |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--local`, `-l` | Run in local dev mode (hot reload, localhost only) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | API host address (ignored in `--local` mode) |
| `PORT` | `8080` | API port |

---

## Testing

### Run All Test Images

```bash
# With GPU
python grounding_dino.py --all

# CPU only
python grounding_dino.py --all --no-gpu
```

### Single Image Test

```bash
python grounding_dino.py
# Uses default: test_photos/images/bedroom1.jpeg
```

### Postman Collection

Import `postman/GroundingDINO.postman_collection.json` for API testing. Use with `postman/images.csv` for batch testing.

---

## GCP Cloud Run Deployment

The service is automatically deployed to Cloud Run when changes are pushed to `main`.

### Recommended Cloud Run Settings

| Setting       | Value  | Notes                      |
|---------------|--------|----------------------------|
| Memory        | 8Gi+   | Model requires ~4GB        |
| CPU           | 4+     | For inference performance  |
| GPU           | 1x L4  | Recommended for production |
| Timeout       | 300s   | First request loads model  |
| Min instances | 0      | Avoid unnecessary uptime   |

---

## Development Notes

### Adding New Object Classes

1. Add class name to `INDOOR_BUSINESS_CLASSES` in `COCO_CLASSES.py`
2. The model will automatically detect the new class (open-vocabulary)
3. Update ground truth annotations as needed

### Creating Test Annotations

Use the annotation tool in `annotation-tool/` in annotation-tool branch or manually create JSON:

```json
{
  "objects": [
    {"class": "bed", "bbox": [x1, y1, x2, y2]},
    {"class": "pillow", "bbox": [x1, y1, x2, y2]}
  ]
}
```

Bounding box format: `[x1, y1, x2, y2]` (top-left and bottom-right corners)

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size or image resolution
- Use CPU mode: `use_gpu=false`
- Ensure no other GPU processes running

### Model Not Loading
- Verify weights file exists: `weights/groundingdino_swint_ogc.pth`
- Check config file: `weights/GroundingDINO_SwinT_OGC.py`
- Ensure GroundingDINO package is installed

### Slow First Request
- Call `/warmup` endpoint after startup
- Set Cloud Run min instances to 0

### Low Detection Accuracy
- Adjust thresholds (lower = more detections, potentially more false positives)
- Verify ground truth annotations are accurate
- Check IoU threshold for matching

---

## Dependencies

Key packages (see `requirements.txt` for full list):

- **PyTorch** >= 2.0.0 (with CUDA for GPU support)
- **FastAPI** + Uvicorn
- **GroundingDINO** (from IDEA-Research)
- **Transformers** (for BERT text encoder)
- **OpenCV** + Pillow (image processing)
