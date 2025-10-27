#!/bin/bash
# setup_env.sh
# Setup Python virtual environment for NeMo-Charlie Trading Pipeline

set -e  # Exit on error

echo "========================================="
echo "NeMo-Charlie Pipeline - Environment Setup"
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

# Install PyTorch
if [[ "${INSTALL_GPU_TORCH:-false}" == "true" ]]; then
    echo "Installing PyTorch with CUDA support (ensure correct CUDA toolkit)..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

if [[ "${INSTALL_NEMO:-false}" == "true" ]]; then
    echo "Installing NeMo-specific requirements..."
    if [[ ! -f requirements_nemo.txt ]]; then
        echo "✗ requirements_nemo.txt not found" >&2
        exit 1
    fi
    pip install -r requirements_nemo.txt
fi

# Verify installations
echo ""
echo "Verifying installations..."

INSTALL_NEMO_VAR="${INSTALL_NEMO:-false}"

python3 << EOF
import sys
import os

def check_import(module_name, display_name=None, required=True):
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {display_name}: {version}")
        return True
    except ImportError as e:
        if required:
            print(f"✗ {display_name}: FAILED - {e}")
        else:
            print(f"○ {display_name}: not installed (optional)")
        return False

print("\nCore dependencies:")
check_import('torch', 'PyTorch')
check_import('transformers', 'Transformers')
check_import('datasets', 'Datasets')

print("\nData processing:")
check_import('pandas', 'Pandas')
check_import('numpy', 'NumPy')
check_import('pyarrow', 'PyArrow')

print("\nUtilities:")
check_import('dotenv', 'python-dotenv')
check_import('yaml', 'PyYAML')
check_import('tqdm', 'tqdm')

# Check NeMo dependencies if INSTALL_NEMO=true
install_nemo = os.environ.get('INSTALL_NEMO_VAR', 'false') == 'true'
if install_nemo:
    print("\nNeMo dependencies:")
    check_import('lightning', 'Lightning')
    check_import('pytorch_lightning', 'PyTorch Lightning')
    check_import('megatron.core', 'Megatron-Core')
    check_import('nemo', 'NeMo')
    check_import('omegaconf', 'OmegaConf')
    check_import('hydra', 'Hydra')

    # Verify NeMo llm module works
    try:
        from nemo.collections import llm
        print("✓ NeMo LLM module: available")
    except Exception as e:
        print(f"✗ NeMo LLM module: FAILED - {e}")

print("\nOptional dependencies:")
check_import('yfinance', 'yfinance', required=False)
check_import('wandb', 'WandB', required=False)
check_import('tensorboard', 'TensorBoard', required=False)

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

if [[ "${INSTALL_NEMO:-false}" == "true" ]]; then
    echo "NeMo Framework installed for DeepSeek-V3 training."
    echo ""
    echo "Next steps:"
    echo "  1. Copy .env.example to .env and configure"
    echo "  2. Run the full pipeline:"
    echo "     bash scripts/run_full_pipeline.sh"
    echo ""
else
    echo "For NeMo training support, re-run with:"
    echo "  INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh"
    echo ""
fi
