#!/bin/bash
# setup_env.sh
# Complete environment setup for NeMo-Charlie Trading Pipeline
# Handles installation, patching, and verification on a fresh machine

set -e  # Exit on error

echo "========================================="
echo "NeMo-Charlie Pipeline - Environment Setup"
echo "========================================="
echo ""
echo "This script will:"
echo "  1. Create Python virtual environment"
echo "  2. Install PyTorch (GPU or CPU)"
echo "  3. Install base dependencies"
echo "  4. Install NeMo Framework (if requested)"
echo "  5. Apply NeMo patches to fix import issues"
echo "  6. Verify all installations"
echo ""

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
echo ""
echo "========================================="
echo "Step 1: Creating Virtual Environment"
echo "========================================="
echo "Location: $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "[WARNING] Virtual environment already exists at $VENV_DIR"
    echo "[WARNING] Continuing will use the existing environment"
    echo "[WARNING] To start fresh, remove it first: rm -rf $VENV_DIR"
    echo ""
else
    python3 -m venv "$VENV_DIR"
    echo "✓ Created virtual environment"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel -q
echo "✓ Build tools upgraded"

# Install PyTorch
echo ""
echo "========================================="
echo "Step 2: Installing PyTorch"
echo "========================================="

if [[ "${INSTALL_GPU_TORCH:-false}" == "true" ]]; then
    echo "Installing PyTorch with CUDA 12.4 support..."
    echo "[INFO] Ensure your system has CUDA 12.4 toolkit installed"
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    echo "✓ PyTorch (GPU) installed"
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    echo "✓ PyTorch (CPU) installed"
fi

# Install base requirements
echo ""
echo "========================================="
echo "Step 3: Installing Base Dependencies"
echo "========================================="

if [[ ! -f requirements.txt ]]; then
    echo "✗ requirements.txt not found" >&2
    exit 1
fi

echo "Installing from requirements.txt..."
pip install -r requirements.txt -q
echo "✓ Base dependencies installed"

# Install NeMo and patch if requested
if [[ "${INSTALL_NEMO:-false}" == "true" ]]; then
    echo ""
    echo "========================================="
    echo "Step 4: Installing NeMo Framework"
    echo "========================================="

    if [[ ! -f requirements_nemo.txt ]]; then
        echo "✗ requirements_nemo.txt not found" >&2
        exit 1
    fi

    # Temporarily disable exit on error for NeMo installation
    set +e

    echo "Installing NeMo and dependencies..."
    echo "[INFO] This may take 5-10 minutes..."
    echo "[INFO] Some optional packages may fail - this is expected"
    echo ""

    pip install -r requirements_nemo.txt
    INSTALL_EXIT_CODE=$?

    # Re-enable exit on error
    set -e

    if [ $INSTALL_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ NeMo requirements installed successfully"
    else
        echo ""
        echo "[WARNING] Some NeMo dependencies failed (exit code: $INSTALL_EXIT_CODE)"
        echo "[WARNING] This is often due to optional packages like:"
        echo "           - mamba-ssm (requires C++ compilation)"
        echo "           - flash-attn (requires CUDA)"
        echo "           - transformer-engine (optional optimization)"
        echo "[INFO] Checking if core NeMo is installed..."
        echo ""
    fi

    # Verify core NeMo is installed
    if python3 -c "import nemo" 2>/dev/null; then
        echo "✓ NeMo core package installed"
    else
        echo "✗ NeMo installation failed completely"
        echo ""
        echo "Try installing manually:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  pip install nemo-toolkit"
        exit 1
    fi

    # Apply NeMo patches
    echo ""
    echo "========================================="
    echo "Step 5: Applying NeMo Patches"
    echo "========================================="
    echo ""
    echo "NeMo 2.5.2 has import issues with optional dependencies:"
    echo "  - nv_one_logger (telemetry - not on public PyPI)"
    echo "  - nemo_run (recipes - not on public PyPI)"
    echo "  - tensorstore (export - optional)"
    echo ""
    echo "Applying patches to make these imports conditional..."
    echo ""

    # Check if patch script exists
    if [[ ! -f scripts/apply_nemo_patches.py ]]; then
        echo "[ERROR] scripts/apply_nemo_patches.py not found"
        echo "[ERROR] Cannot apply NeMo patches"
        echo ""
        echo "Please ensure you have the latest code:"
        echo "  git pull origin main"
        exit 1
    fi

    # Run the patch script
    python scripts/apply_nemo_patches.py
    PATCH_EXIT_CODE=$?

    if [ $PATCH_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ NeMo patches applied successfully"
    else
        echo ""
        echo "[ERROR] Failed to apply NeMo patches (exit code: $PATCH_EXIT_CODE)"
        echo "[ERROR] NeMo imports may not work"
        echo ""
        echo "You can try applying patches manually:"
        echo "  source $VENV_DIR/bin/activate"
        echo "  python scripts/apply_nemo_patches.py"
        echo ""
        echo "Or see NEMO_FIXES.md for manual patching instructions"
        exit 1
    fi
else
    echo ""
    echo "========================================="
    echo "Step 4-5: Skipped (NeMo not requested)"
    echo "========================================="
    echo ""
    echo "To install NeMo, re-run with:"
    echo "  INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh"
fi

# Comprehensive verification
echo ""
echo "========================================="
echo "Step 6: Verifying Installation"
echo "========================================="
echo ""

INSTALL_NEMO_VAR="${INSTALL_NEMO:-false}"

python3 << 'EOF'
import sys
import os

def check_import(module_name, display_name=None, required=True):
    """Check if a module can be imported and display status"""
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
        return True
    except ImportError as e:
        if required:
            print(f"  ✗ {display_name}: FAILED - {e}")
        else:
            print(f"  ○ {display_name}: not installed (optional)")
        return False

print("Core dependencies:")
all_ok = True
all_ok &= check_import('torch', 'PyTorch')
all_ok &= check_import('transformers', 'Transformers')
all_ok &= check_import('datasets', 'Datasets')

print("\nData processing:")
all_ok &= check_import('pandas', 'Pandas')
all_ok &= check_import('numpy', 'NumPy')
all_ok &= check_import('pyarrow', 'PyArrow')

print("\nUtilities:")
all_ok &= check_import('dotenv', 'python-dotenv')
all_ok &= check_import('yaml', 'PyYAML')
all_ok &= check_import('tqdm', 'tqdm')

# Check NeMo dependencies if requested
install_nemo = os.environ.get('INSTALL_NEMO_VAR', 'false') == 'true'
if install_nemo:
    print("\nNeMo Framework:")
    all_ok &= check_import('lightning', 'Lightning')
    all_ok &= check_import('pytorch_lightning', 'PyTorch Lightning')
    all_ok &= check_import('megatron.core', 'Megatron-Core')
    all_ok &= check_import('nemo', 'NeMo')
    all_ok &= check_import('omegaconf', 'OmegaConf')
    all_ok &= check_import('hydra', 'Hydra')

    print("\nNeMo LLM Module (Critical Test):")
    try:
        from nemo.collections import llm
        print("  ✓ nemo.collections.llm: importable")

        # Check for critical functions
        if hasattr(llm, 'import_ckpt'):
            print("  ✓ llm.import_ckpt: available")
        else:
            print("  ✗ llm.import_ckpt: NOT AVAILABLE")
            all_ok = False

        if hasattr(llm, 'DeepSeekV3Config'):
            print("  ✓ llm.DeepSeekV3Config: available")
        else:
            print("  ✗ llm.DeepSeekV3Config: NOT AVAILABLE")
            all_ok = False

        if hasattr(llm, 'DeepSeekModel'):
            print("  ✓ llm.DeepSeekModel: available")
        else:
            print("  ✗ llm.DeepSeekModel: NOT AVAILABLE")
            all_ok = False

    except Exception as e:
        print(f"  ✗ nemo.collections.llm: FAILED - {e}")
        print("\n[ERROR] NeMo LLM import failed!")
        print("[ERROR] This means the patches were not applied correctly.")
        print("[ERROR] Try running: python scripts/apply_nemo_patches.py")
        all_ok = False

print("\nOptional dependencies:")
check_import('yfinance', 'yfinance', required=False)
check_import('wandb', 'WandB', required=False)
check_import('tensorboard', 'TensorBoard', required=False)

# Exit with error if any required import failed
if not all_ok:
    print("\n[ERROR] Some required dependencies are missing or not working")
    sys.exit(1)

sys.exit(0)
EOF

VERIFICATION_EXIT_CODE=$?

echo ""
if [ $VERIFICATION_EXIT_CODE -eq 0 ]; then
    echo "✓ All verifications passed"
else
    echo "✗ Verification failed - see errors above"
    exit 1
fi

# Final summary
echo ""
echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Python: $(which python)"
echo ""

if [[ "${INSTALL_NEMO:-false}" == "true" ]]; then
    echo "✓ NeMo Framework is installed and patched"
    echo "✓ Ready for DeepSeek-V3 model conversion and training"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "       source $VENV_DIR/bin/activate"
    echo ""
    echo "  2. Configure your environment:"
    echo "       cp .env.example .env"
    echo "       # Edit .env with your settings"
    echo ""
    echo "  3. Test NeMo conversion script:"
    echo "       python scripts/convert/import_to_nemo.py --help"
    echo ""
    echo "  4. Run the full pipeline:"
    echo "       bash scripts/run_full_pipeline.sh"
    echo ""
    echo "To verify patches are still working later:"
    echo "  bash scripts/verify_nemo_fixes.sh"
    echo ""
else
    echo "Base dependencies installed (no NeMo)"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment:"
    echo "       source $VENV_DIR/bin/activate"
    echo ""
    echo "  2. Run data processing:"
    echo "       python src/parsers/xml_to_jsonl.py"
    echo ""
    echo "To install NeMo later, run:"
    echo "  INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh"
    echo ""
fi

echo "To deactivate the environment later:"
echo "  deactivate"
echo ""
