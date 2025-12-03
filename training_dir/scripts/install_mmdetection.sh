#!/bin/bash

# Install MMDetection with Grounding DINO support

# Don't exit on error - we want to try multiple installation methods
set +e

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
echo "MMDetection Installation Script"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Python version: $PYTHON_VERSION"

# Warn about Python 3.13+ compatibility
if [ "$PYTHON_MAJOR" -gt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]); then
    echo "⚠️  WARNING: Python 3.13+ may have compatibility issues with mmcv"
    echo "   Recommended: Python 3.10 or 3.11"
    echo "   Continuing anyway, but you may encounter build errors..."
    echo ""
fi

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    CUDA_AVAILABLE=true
else
    echo "⚠️  CUDA not detected, will install CPU-only version"
    CUDA_AVAILABLE=false
fi

# Upgrade pip and setuptools
echo ""
echo "Upgrading pip and setuptools..."
$PIP_CMD install --upgrade pip setuptools wheel

# Install openmim first
echo ""
echo "Installing openmim..."
$PIP_CMD install -U openmim

# Check if mmdetection directory exists
if [ ! -d "mmdetection" ]; then
    echo ""
    echo "Cloning MMDetection repository..."
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    git checkout v3.1.0  # Use stable version
else
    echo ""
    echo "MMDetection directory exists, checking out v3.1.0..."
    cd mmdetection
    git fetch
    git checkout v3.1.0 || git checkout main
fi

# Install PyTorch first (required for mmdetection build)
echo ""
echo "Installing PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision || {
        echo "⚠️  CUDA PyTorch installation failed, installing CPU version..."
        $PIP_CMD install torch torchvision
    }
else
    echo "Installing PyTorch (CPU version)..."
    $PIP_CMD install torch torchvision
fi

# Install mmcv - must be <2.1.0 for MMDetection v3.1.0
echo ""
echo "Installing MMCV (this may take several minutes)..."
echo "Note: MMDetection v3.1.0 requires mmcv>=2.0.0rc4,<2.1.0"

# Method 1: Try using mim with specific version constraint
echo "Attempting to install mmcv 2.0.x via mim..."
if mim install "mmcv>=2.0.0rc4,<2.1.0" --timeout 60; then
    echo "✅ MMCV installed via mim"
else
    echo "⚠️  mim install failed, trying alternative method..."
    
    # Method 2: Try installing specific version directly
    echo "Attempting to install mmcv 2.0.1..."
    $PIP_CMD install "mmcv>=2.0.0rc4,<2.1.0" --no-build-isolation || {
        echo "⚠️  Specific version failed, trying latest 2.0.x..."
        $PIP_CMD install "mmcv==2.0.1" --no-build-isolation || \
        $PIP_CMD install "mmcv==2.0.0" --no-build-isolation || \
        echo "⚠️  MMCV installation had issues, but continuing..."
    }
fi

# Install mmengine
echo ""
echo "Installing mmengine..."
$PIP_CMD install "mmengine>=0.8.0"

# Upgrade setuptools and build tools (fixes build_editable issues)
echo ""
echo "Upgrading build tools..."
$PIP_CMD install --upgrade pip setuptools>=65.0.0 wheel build

# Install mmdetection in development mode
echo ""
echo "Installing MMDetection (this may take several minutes)..."
echo "Note: This requires PyTorch to be installed first"

# Install dependencies first
if [ -f "requirements/runtime.txt" ]; then
    echo "Installing runtime dependencies..."
    $PIP_CMD install -r requirements/runtime.txt
fi
if [ -f "requirements/build.txt" ]; then
    echo "Installing build dependencies..."
    $PIP_CMD install -r requirements/build.txt
fi

# Try installing in editable mode
echo "Attempting editable installation..."
if $PIP_CMD install -e . --no-build-isolation 2>/dev/null; then
    echo "✅ MMDetection installed successfully in editable mode"
else
    echo "⚠️  Editable installation failed (build_editable hook issue)"
    echo "    Trying non-editable installation..."
    # Try regular installation (non-editable)
    if $PIP_CMD install . --no-build-isolation; then
        echo "✅ MMDetection installed successfully (non-editable mode)"
        echo "   Note: To update, reinstall: cd mmdetection && $PIP_CMD install . --upgrade"
    else
        echo "⚠️  Installation had issues"
        echo "   Try running: bash scripts/install_mmdet_non_editable.sh"
    fi
fi

# Install additional dependencies for Grounding DINO
echo ""
echo "Installing Grounding DINO dependencies..."
$PIP_CMD install transformers sentencepiece

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="

INSTALLATION_SUCCESS=true

python3 -c "import mmcv; print(f'✅ mmcv version: {mmcv.__version__}')" || {
    echo "❌ mmcv import failed"
    INSTALLATION_SUCCESS=false
}

python3 -c "import mmdet; print(f'✅ mmdet version: {mmdet.__version__}')" || {
    echo "❌ mmdet import failed"
    INSTALLATION_SUCCESS=false
}

python3 -c "import mmengine; print(f'✅ mmengine version: {mmengine.__version__}')" || {
    echo "❌ mmengine import failed"
    INSTALLATION_SUCCESS=false
}

echo ""
echo "=========================================="
if [ "$INSTALLATION_SUCCESS" = true ]; then
    echo "✅ Installation complete!"
    echo "=========================================="
    echo ""
    echo "Next step: Download Grounding DINO model weights using scripts/download_model.sh"
    exit 0
else
    echo "❌ Installation had issues"
    echo "=========================================="
    echo ""
    echo "Troubleshooting steps:"
    echo "1. Try using Python 3.10 or 3.11 (recommended)"
    echo "   See INSTALL_PYTHON310.md for instructions"
    echo ""
    echo "2. Try standalone mmcv installation:"
    echo "   bash scripts/install_mmcv_standalone.sh"
    echo ""
    echo "3. Check that you have build tools installed:"
    echo "   - macOS: xcode-select --install"
    echo "   - Ubuntu: sudo apt-get install build-essential"
    echo ""
    echo "4. See TROUBLESHOOTING.md for more solutions"
    echo ""
    exit 1
fi

