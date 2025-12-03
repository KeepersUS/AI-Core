#!/bin/bash
# Generate pseudo labels using OFFICIAL Grounding DINO with existing weights
# Uses system Python (where Grounding DINO package is installed)

cd "$(dirname "$0")/.."

echo "🚀 Starting pseudo-label generation with OFFICIAL Grounding DINO..."
echo "📊 Total images to process: 544"
echo "⏱️  Estimated time: 25-30 minutes"
echo ""
echo "Using system Python (where Grounding DINO package is installed)"
echo ""

# Use system Python if venv is active
if [ -n "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment detected. Using system Python instead..."
    SYSTEM_PYTHON="/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"
    if [ ! -f "$SYSTEM_PYTHON" ]; then
        echo "Deactivating venv and using default python3..."
        deactivate 2>/dev/null || true
        SYSTEM_PYTHON="python3"
    fi
else
    SYSTEM_PYTHON="python3"
fi

PROMPT="bed . chair . couch . dining table . toilet . sink . refrigerator . oven . microwave . tv . laptop . keyboard . mouse . bottle . cup . bowl . book . clock . lamp . pillow . towel . mirror . door . window . person . backpack . handbag . suitcase . wine glass . fork . knife . spoon . potted plant . remote . cell phone . toaster . vase . scissors . dining chair . nightstand . dresser"

$SYSTEM_PYTHON scripts/generate_pseudo_labels_official_weights.py \
  --prompt "$PROMPT" \
  --threshold 0.3 \
  --max-detections 20 \
  --box-threshold 0.3 \
  --text-threshold 0.3 \
  --output data/annotations/pseudo_labels_official.json

echo ""
echo "✅ Done! Results saved to: data/annotations/pseudo_labels_official.json"
echo ""
echo "📊 Validate results with:"
echo "python3 scripts/comprehensive_label_validation.py --labels data/annotations/pseudo_labels_official.json"
