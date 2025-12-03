#!/bin/bash

# Build mmcv from source with extensions for CPU

set +e

# Determine pip command
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    PIP_CMD="pip"
else
    PIP_CMD="python3 -m pip"
fi

echo "=========================================="
echo "Building MMCV from Source"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This will build mmcv from source"
echo "   This may take 15-30 minutes on macOS"
echo "   You need build tools installed (xcode-select --install)"
echo ""

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in virtual environment"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/venv/bin/activate" || {
        echo "❌ Could not activate virtual environment"
        exit 1
    }
fi

echo "✅ Virtual environment: $VIRTUAL_ENV"
echo ""

# Check build tools
echo "Checking build tools..."
if ! command -v clang &> /dev/null; then
    echo "⚠️  clang not found"
    echo "   Install Xcode Command Line Tools: xcode-select --install"
    exit 1
fi

# Check PyTorch
echo "Checking PyTorch..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch..."
    $PIP_CMD install torch torchvision
fi

python3 -c "import torch; print(f'✅ PyTorch: {torch.__version__}')"

# Uninstall existing mmcv
echo ""
echo "Uninstalling existing mmcv..."
$PIP_CMD uninstall mmcv mmcv-full -y

# Install build dependencies
echo ""
echo "Installing build dependencies..."
$PIP_CMD install --upgrade pip setuptools wheel
$PIP_CMD install cython numpy

# Clone mmcv
echo ""
echo "Cloning mmcv repository..."
MMCV_DIR="/tmp/mmcv_build"
rm -rf "$MMCV_DIR"
git clone https://github.com/open-mmlab/mmcv.git "$MMCV_DIR"
cd "$MMCV_DIR"
git checkout v2.0.1

# Build mmcv
echo ""
echo "Building mmcv from source (this will take 15-30 minutes)..."
echo "⚠️  This may use significant CPU and memory"

# Set environment variables for CPU-only build
export MMCV_WITH_OPS=1
export MAX_JOBS=4  # Limit parallel jobs

# Install mmcv in development mode (builds extensions)
$PIP_CMD install -e . -v || {
    echo "⚠️  Development install failed, trying regular install..."
    $PIP_CMD install . -v
}

# Cleanup
cd - > /dev/null
rm -rf "$MMCV_DIR"

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

if python3 -c "import mmcv; print(f'✅ mmcv version: {mmcv.__version__}')" 2>/dev/null; then
    echo "✅ mmcv imported"
    
    # Try to import _ext
    if python3 -c "from mmcv import _ext; print('✅ mmcv._ext found!')" 2>/dev/null; then
        echo "✅ mmcv._ext found!"
        echo ""
        echo "✅ Installation successful!"
        exit 0
    else
        echo "⚠️  mmcv._ext not found (may still work for some operations)"
    fi
else
    echo "❌ mmcv import failed"
    exit 1
fi

echo ""
echo "Note: If mmcv._ext is still not found, you may need to:"
echo "1. Install more build dependencies"
echo "2. Use a different Python version"
echo "3. Try installing on a system with CUDA"



