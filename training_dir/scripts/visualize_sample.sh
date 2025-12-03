#!/bin/bash
# Quick script to visualize a sample of pseudo-labels

cd "$(dirname "$0")/.."

echo "🎨 Visualizing sample pseudo-labels..."
echo ""

# Visualize 10 random images
python3 scripts/visualize_official_labels.py \
  --labels "data/annotations/pseudo_labels_official.json" \
  --num-samples 10 \
  --output-dir "data/visualizations_official"

echo ""
echo "✅ Visualizations saved to: data/visualizations_official/"
echo ""
echo "💡 Check the images to verify box placement and label accuracy!"

