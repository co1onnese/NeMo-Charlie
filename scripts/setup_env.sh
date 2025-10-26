#!/bin/bash
# setup_env.sh
# Setup Python virtual environment for SFT Trading Pipeline

set -e  # Exit on error

echo "========================================="
echo "SFT Trading Pipeline - Environment Setup"
echo "========================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$REQUIRED_VERSION'.split('.'))) else 1)"; then
    echo "✓ Python version is sufficient"
else
    echo "✗ Python $REQUIRED_VERSION or higher is required"
    exit 1
fi

# Create virtual environment
VENV_DIR="${VENV_DIR:-venv}"
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version for initial testing)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installations
echo ""
echo "Verifying installations..."

python3 << 'EOF'
import sys

def check_import(module_name, display_name=None):
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError as e:
        print(f"✗ {display_name}: FAILED - {e}")
        return False

print("\nCore dependencies:")
check_import('torch', 'PyTorch')
check_import('transformers', 'Transformers')
check_import('datasets', 'Datasets')
check_import('trl', 'TRL')
check_import('peft', 'PEFT')

print("\nData processing:")
check_import('pandas', 'Pandas')
check_import('numpy', 'NumPy')

print("\nUtilities:")
check_import('dotenv', 'python-dotenv')
check_import('yaml', 'PyYAML')
check_import('tqdm', 'tqdm')

print("\nOptional:")
try:
    import yfinance
    print(f"✓ yfinance: {yfinance.__version__}")
except:
    print("○ yfinance: not installed (optional)")

try:
    import wandb
    print(f"✓ wandb: {wandb.__version__}")
except:
    print("○ wandb: not installed (optional)")

EOF

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the environment, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
