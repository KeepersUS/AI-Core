#!/bin/bash

# Download Grounding DINO pre-trained model weights

set -e

MODEL_DIR="checkpoints"
MODEL_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
MODEL_NAME="groundingdino_swint_ogc.pth"

# Create checkpoints directory if it doesn't exist
mkdir -p $MODEL_DIR

# Check if model already exists
if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Model already exists at $MODEL_DIR/$MODEL_NAME"
    echo "Skipping download. Delete the file to re-download."
    exit 0
fi

echo "Downloading Grounding DINO model..."
echo "URL: $MODEL_URL"
echo "Destination: $MODEL_DIR/$MODEL_NAME"

# Download using wget or curl
if command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/$MODEL_NAME" "$MODEL_URL"
else
    echo "Error: Neither wget nor curl found. Please install one of them."
    echo "Or manually download from: $MODEL_URL"
    exit 1
fi

if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "✅ Model downloaded successfully!"
    echo "Location: $MODEL_DIR/$MODEL_NAME"
else
    echo "❌ Download failed. Please download manually from: $MODEL_URL"
    exit 1
fi


