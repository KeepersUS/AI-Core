#!/bin/bash

# Standalone script to install mmcv with better error handling
# Use this if the main installation script fails

set -e

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    # Fall back to python3 -m pip (most reliable)
    PIP_CMD="python3 -m pip"
fi

echo "Using: $PIP_CMD"

echo "=========================================="
echo "MMCV Standalone Installation"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python version: $PYTHON_VERSION"

# Check CUDA version
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    echo "CUDA version: $CUDA_VERSION"
    
    # Determine PyTorch CUDA version
    TORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
    echo "PyTorch CUDA: $TORCH_CUDA"
else
    echo "CUDA not detected - installing CPU version"
    TORCH_CUDA="cpu"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel

# Install openmim
echo ""
echo "Installing openmim..."
$PIP_CMD install -U openmim

# Try different installation methods
echo ""
echo "Attempting to install mmcv..."
echo "Note: For MMDetection v3.1.0, need mmcv>=2.0.0rc4,<2.1.0"

# Method 1: Use mim with correct version (recommended)
echo "Method 1: Using mim with version constraint..."
if mim install "mmcv>=2.0.0rc4,<2.1.0" --timeout 120; then
    echo "✅ Successfully installed mmcv via mim"
    exit 0
fi

# Method 2: Install from pre-built wheels based on CUDA version
if [ "$TORCH_CUDA" != "none" ] && [ "$TORCH_CUDA" != "cpu" ]; then
    echo ""
    echo "Method 2: Installing from pre-built wheels (CUDA $TORCH_CUDA)..."
    
    # Try different CUDA versions
    for CUDA_VER in "cu121" "cu118" "cu117" "cu116"; do
        echo "Trying CUDA $CUDA_VER..."
        if $PIP_CMD install mmcv -f "https://download.openmmlab.com/mmcv/dist/$CUDA_VER/torch2.0/index.html" 2>/dev/null; then
            echo "✅ Successfully installed mmcv for CUDA $CUDA_VER"
            exit 0
        fi
    done
fi

# Method 3: Install CPU version
echo ""
echo "Method 3: Installing CPU version..."
if $PIP_CMD install "mmcv>=2.0.0rc4,<2.1.0" --no-build-isolation; then
    echo "✅ Successfully installed mmcv (CPU)"
    exit 0
fi

# Method 4: Try specific version
echo ""
echo "Method 4: Installing mmcv 2.0.1 (may take longer)..."
if $PIP_CMD install "mmcv==2.0.1" --no-build-isolation; then
    echo "✅ Successfully installed mmcv 2.0.1"
    exit 0
fi

# If all methods fail
echo ""
echo "❌ All installation methods failed"
echo ""
echo "Troubleshooting:"
echo "1. Check Python version (recommended: 3.10 or 3.11)"
echo "2. Install build dependencies:"
echo "   - macOS: xcode-select --install"
echo "   - Ubuntu: sudo apt-get install build-essential"
echo "3. Try creating a new environment with Python 3.10:"
echo "   conda create -n dino_train python=3.10"
echo "   conda activate dino_train"
echo ""
exit 1

