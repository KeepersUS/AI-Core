#!/bin/bash

# Install Python 3.10 on macOS without conda
# This script provides multiple options

set -e

echo "=========================================="
echo "Python 3.10 Installation for macOS"
echo "=========================================="
echo ""

# Check current Python version
CURRENT_PYTHON=$(python3 --version 2>&1)
echo "Current Python: $CURRENT_PYTHON"
echo ""

# Option 1: Check if pyenv is available
if command -v pyenv &> /dev/null; then
    echo "✅ pyenv is installed"
    echo ""
    echo "To install Python 3.10 with pyenv:"
    echo "  1. pyenv install 3.10.12"
    echo "  2. cd $(pwd)"
    echo "  3. pyenv local 3.10.12"
    echo "  4. python -m venv venv"
    echo "  5. source venv/bin/activate"
    echo ""
    read -p "Do you want to install Python 3.10 with pyenv now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Python 3.10.12 with pyenv..."
        pyenv install 3.10.12
        pyenv local 3.10.12
        echo "✅ Python 3.10.12 installed!"
        echo "   Current directory is now using Python 3.10.12"
        echo "   Run: python -m venv venv && source venv/bin/activate"
        exit 0
    fi
else
    echo "❌ pyenv is not installed"
    echo ""
    echo "Option 1: Install pyenv (recommended)"
    echo "  brew install pyenv"
    echo "  Then add to ~/.zshrc:"
    echo "    export PYENV_ROOT=\"\$HOME/.pyenv\""
    echo "    export PATH=\"\$PYENV_ROOT/bin:\$PATH\""
    echo "    eval \"\$(pyenv init -)\""
    echo "  Then run this script again"
    echo ""
fi

# Option 2: Check if Homebrew is available
if command -v brew &> /dev/null; then
    echo "✅ Homebrew is installed"
    echo ""
    echo "Option 2: Install Python 3.10 with Homebrew"
    echo "  brew install python@3.10"
    echo "  /opt/homebrew/bin/python3.10 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Do you want to install Python 3.10 with Homebrew now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Python 3.10 with Homebrew..."
        brew install python@3.10
        /opt/homebrew/bin/python3.10 -m venv venv
        echo "✅ Python 3.10 installed and virtual environment created!"
        echo "   Run: source venv/bin/activate"
        exit 0
    fi
else
    echo "❌ Homebrew is not installed"
    echo ""
    echo "Option 2: Install Homebrew first"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  Then install Python 3.10:"
    echo "    brew install python@3.10"
    echo ""
fi

# Option 3: Manual download
echo "Option 3: Manual installation"
echo "  1. Download Python 3.10 from: https://www.python.org/downloads/release/python-31012/"
echo "  2. Install the .pkg file"
echo "  3. Create virtual environment:"
echo "     /usr/local/bin/python3.10 -m venv venv"
echo "     source venv/bin/activate"
echo ""

# Option 4: Try to proceed with current Python (if it's close enough)
echo "Option 4: Continue with current Python"
echo "  You can try to continue with your current Python version"
echo "  The installation script will attempt to work around compatibility issues"
echo ""

read -p "Do you want to continue with the current Python and try installation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Proceeding with installation..."
    echo "If you encounter errors, try one of the options above to install Python 3.10"
    echo ""
    exit 0
else
    echo ""
    echo "Please choose one of the installation options above."
    echo "See INSTALL_PYTHON310.md for detailed instructions."
    exit 1
fi



