#!/bin/bash
# setup_env_v2.sh - Production-ready NeMo environment setup
# 
# Optimized installation order to minimize warnings:
#   1. PyTorch (foundation)
#   2. Optional deps BEFORE NeMo (zarr, transformer-engine)
#   3. Base requirements
#   4. NeMo (detects optional deps already installed)
#   5. Patches
#
# Reads CPU_ONLY_MODE from .env to determine GPU vs CPU setup
# Fails if venv/ already exists (clean slate required)

set -e  # Exit immediately on any error

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"
ENV_FILE="$PROJECT_ROOT/.env"

# ============================================================================
# Helper Functions
# ============================================================================

banner() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

error_exit() {
    echo "" >&2
    echo "ERROR: $1" >&2
    echo "" >&2
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        error_exit "$1 is not installed. Please install it first."
    fi
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

banner "NeMo-Charlie Setup v2 - Production Environment"

echo "Project root: $PROJECT_ROOT"
echo "Target venv: $VENV_DIR"
echo ""

# Check for clean slate
if [ -d "$VENV_DIR" ]; then
    error_exit "Virtual environment already exists at $VENV_DIR
    
    This script requires a clean slate. To start fresh:
      rm -rf $VENV_DIR
      bash scripts/setup_env_v2.sh"
fi

# Check required commands
echo "Checking prerequisites..."
check_command python3.12
check_command git

# Check Python version
PYTHON_VERSION=$(python3.12 --version 2>&1 | awk '{print $2}')
echo "✓ Python: $PYTHON_VERSION"

# Verify Python 3.12+
if ! python3.12 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    error_exit "Python 3.12+ required, found $PYTHON_VERSION"
fi

# Check for .env file
if [ ! -f "$ENV_FILE" ]; then
    error_exit ".env file not found at $ENV_FILE
    
    Please copy .env.example to .env:
      cp .env.example .env
      # Edit .env and set CPU_ONLY_MODE=true or false"
fi

# Read CPU_ONLY_MODE from .env
if grep -q "^CPU_ONLY_MODE=true" "$ENV_FILE"; then
    CPU_ONLY_MODE=true
    echo "✓ Mode: CPU-only (from .env)"
elif grep -q "^CPU_ONLY_MODE=false" "$ENV_FILE"; then
    CPU_ONLY_MODE=false
    echo "✓ Mode: GPU-enabled (from .env)"
else
    error_exit "CPU_ONLY_MODE not found or invalid in .env
    
    Please add to .env:
      CPU_ONLY_MODE=true   # for CPU-only development
      CPU_ONLY_MODE=false  # for GPU servers"
fi

# GPU mode: check CUDA availability
if [ "$CPU_ONLY_MODE" = false ]; then
    echo ""
    echo "GPU mode selected - checking CUDA..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+")
        echo "✓ CUDA detected: $CUDA_VERSION"
        
        # Verify CUDA 12.x
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        if [ "$CUDA_MAJOR" != "12" ]; then
            error_exit "CUDA 12.x required for PyTorch 2.9.0+cu128, found $CUDA_VERSION
            
            Please install CUDA 12.x or set CPU_ONLY_MODE=true in .env"
        fi
    else
        error_exit "GPU mode selected but nvidia-smi not found
        
        Either:
          1. Install NVIDIA drivers and CUDA 12.x
          2. Set CPU_ONLY_MODE=true in .env for CPU-only setup"
    fi
fi

# Check disk space (need at least 10GB, 20GB for GPU)
REQUIRED_SPACE_GB=$( [ "$CPU_ONLY_MODE" = true ] && echo "10" || echo "20" )
AVAILABLE_GB=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -lt "$REQUIRED_SPACE_GB" ]; then
    error_exit "Insufficient disk space. Required: ${REQUIRED_SPACE_GB}GB, Available: ${AVAILABLE_GB}GB"
fi
echo "✓ Disk space: ${AVAILABLE_GB}GB available (${REQUIRED_SPACE_GB}GB required)"

echo ""
echo "All prerequisites met. Starting installation..."
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================================
# Step 1: Create Virtual Environment
# ============================================================================

banner "Step 1/7: Creating Virtual Environment"

python3.12 -m venv "$VENV_DIR"
echo "✓ Virtual environment created at $VENV_DIR"

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q
echo "✓ Build tools upgraded"

# ============================================================================
# Step 2: Install PyTorch
# ============================================================================

banner "Step 2/7: Installing PyTorch"

if [ "$CPU_ONLY_MODE" = true ]; then
    echo "Installing PyTorch (CPU version)..."
    pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cpu
    echo "✓ PyTorch 2.9.0 (CPU) installed"
else
    echo "Installing PyTorch 2.9.0+cu128 (GPU version)..."
    echo "Download size: ~2GB, this may take a few minutes..."
    pip install torch==2.9.0 torchvision --index-url https://download.pytorch.org/whl/cu128
    echo "✓ PyTorch 2.9.0+cu128 (GPU) installed"
fi

# Verify PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# ============================================================================
# Step 3: Install Optional Dependencies BEFORE NeMo
# ============================================================================

banner "Step 3/7: Installing Optional Dependencies (Before NeMo)"

echo "This step prevents warnings by installing optional deps before NeMo..."
echo ""

# zarr - always install (no compilation, works on CPU and GPU)
# Note: NeMo 2.5.2 requires zarr 2.x, not 3.x
echo "[1/2] Installing zarr (checkpoint format support)..."
pip install "zarr>=2.16.0,<3.0.0" -q
echo "✓ zarr installed"

# transformer-engine - GPU only
if [ "$CPU_ONLY_MODE" = false ]; then
    echo ""
    echo "[2/2] Installing transformer-engine (GPU performance optimization)..."
    echo "Download size: ~500MB, this may take a few minutes..."
    
    # Try to install transformer-engine
    if pip install transformer-engine[pytorch] -q; then
        echo "✓ transformer-engine installed"
    else
        echo "⚠ transformer-engine installation failed (non-critical)"
        echo "  Continuing without it - training will still work"
    fi
else
    echo ""
    echo "[2/2] Skipping transformer-engine (CPU-only mode)"
fi

echo ""
echo "✓ Optional dependencies installed"

# ============================================================================
# Step 4: Install Base Requirements
# ============================================================================

banner "Step 4/7: Installing Base Requirements"

if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
    error_exit "requirements.txt not found in $PROJECT_ROOT"
fi

echo "Installing from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt" -q
echo "✓ Base requirements installed"

# ============================================================================
# Step 5: Install NeMo and Dependencies
# ============================================================================

banner "Step 5/7: Installing NeMo Framework"

if [ ! -f "$PROJECT_ROOT/requirements_nemo.txt" ]; then
    error_exit "requirements_nemo.txt not found in $PROJECT_ROOT"
fi

echo "Installing NeMo and dependencies..."
echo "This may take 5-10 minutes and will show warnings about optional packages."
echo "These warnings are expected and will be handled by patches in the next step."
echo ""

# Install NeMo (allow warnings, don't fail on optional package errors)
set +e
pip install -r "$PROJECT_ROOT/requirements_nemo.txt"
NEMO_EXIT_CODE=$?
set -e

# Verify core NeMo installed
if python -c "import nemo" 2>/dev/null; then
    NEMO_VERSION=$(python -c "import nemo; print(nemo.__version__)")
    echo ""
    echo "✓ NeMo $NEMO_VERSION installed successfully"
else
    error_exit "NeMo installation failed. Check the error messages above."
fi

# ============================================================================
# Step 6: Apply NeMo Patches
# ============================================================================

banner "Step 6/7: Applying NeMo Patches"

echo "NeMo 2.5.2 requires patches for optional dependencies:"
echo "  - nv_one_logger (telemetry - not on public PyPI)"
echo "  - nemo_run (recipes - not on public PyPI)"
echo ""

if [ ! -f "$PROJECT_ROOT/scripts/apply_nemo_patches.py" ]; then
    error_exit "Patch script not found: scripts/apply_nemo_patches.py"
fi

python "$PROJECT_ROOT/scripts/apply_nemo_patches.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ NeMo patches applied successfully"
else
    error_exit "Failed to apply NeMo patches. Check errors above."
fi

# ============================================================================
# Step 7: Comprehensive Validation
# ============================================================================

banner "Step 7/7: Comprehensive Validation"

echo "Running validation script..."
echo ""

# Run the comprehensive validation script
python "$PROJECT_ROOT/scripts/validate_environment.py"
VALIDATION_EXIT=$?

if [ $VALIDATION_EXIT -eq 0 ]; then
    echo ""
    banner "✓ Setup Complete and Validated!"
else
    error_exit "Validation failed. Check errors above."
fi

# Quick inline verification (backup)
python << 'VERIFY_EOF'
import sys

def test_import(module_name, display_name=None):
    """Test if module imports correctly"""
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
        return True
    except Exception as e:
        print(f"  ✗ {display_name}: FAILED - {e}")
        return False

print("Core packages:")
all_ok = True
all_ok &= test_import('torch', 'PyTorch')
all_ok &= test_import('transformers', 'Transformers')
all_ok &= test_import('datasets', 'Datasets')
all_ok &= test_import('pandas', 'Pandas')

print("\nOptional packages:")
test_import('zarr', 'Zarr')

print("\nNeMo Framework:")
all_ok &= test_import('nemo', 'NeMo')
all_ok &= test_import('megatron.core', 'Megatron-Core')

print("\nNeMo LLM Module (Critical Test):")
try:
    from nemo.collections import llm
    print("  ✓ nemo.collections.llm: importable")
    
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
        
except Exception as e:
    print(f"  ✗ NeMo LLM import failed: {e}")
    all_ok = False

if not all_ok:
    sys.exit(1)
    
sys.exit(0)
VERIFY_EOF

# Success summary already printed by validate_environment.py
echo ""
echo "Virtual environment: $VENV_DIR"
echo "Mode: $([ "$CPU_ONLY_MODE" = true ] && echo "CPU-only" || echo "GPU-enabled")"
echo ""
echo "To activate this environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test model conversion:"
echo "  python scripts/convert/import_to_nemo.py --help"
echo ""
echo "To run the full pipeline:"
echo "  bash scripts/run_full_pipeline.sh"
echo ""
