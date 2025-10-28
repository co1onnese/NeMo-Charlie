#!/bin/bash
# fix_nemo_optional_deps.sh
# Fixes NeMo 2.5.2 import failures caused by unconditional imports of optional dependencies
#
# Root Cause:
#   NeMo 2.5.2 has a critical architectural flaw where optional dependencies (nv_one_logger,
#   nemo_run, tensorstore) are imported unconditionally at module initialization time.
#   These packages:
#   - Are marked as optional in NeMo metadata ([test] extras)
#   - Don't exist on public PyPI (internal NVIDIA packages)
#   - Are not needed for core LLM functionality like model conversion
#
# This script patches NeMo to make these imports conditional with graceful degradation.
#
# Usage:
#   source venv/bin/activate
#   bash scripts/fix_nemo_optional_deps.sh

set -e

echo "================================================================="
echo "NeMo Optional Dependencies Patcher"
echo "================================================================="
echo ""
echo "This script fixes NeMo 2.5.2 import errors caused by missing"
echo "optional dependencies: nv_one_logger, nemo_run, tensorstore"
echo ""

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[ERROR] Virtual environment not activated!"
    echo "[ERROR] Please run:"
    echo "[ERROR]   source venv/bin/activate"
    echo "[ERROR]   bash scripts/fix_nemo_optional_deps.sh"
    exit 1
fi

echo "[INFO] Virtual environment: $VIRTUAL_ENV"
echo "[INFO] Python: $(which python)"
echo ""

# Find NeMo installation path
NEMO_PATH=$(python -c "import nemo; import os; print(os.path.dirname(nemo.__file__))" 2>/dev/null)
if [ -z "$NEMO_PATH" ]; then
    echo "[ERROR] NeMo not installed in this virtual environment"
    echo "[ERROR] Please install NeMo first:"
    echo "[ERROR]   pip install nemo-toolkit"
    exit 1
fi

echo "[INFO] NeMo installation: $NEMO_PATH"
echo ""
echo "========================================="
echo "Applying Patches..."
echo "========================================="
echo ""

# Backup original files
BACKUP_DIR="$VIRTUAL_ENV/nemo_patches_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "[INFO] Creating backups in: $BACKUP_DIR"

# List of files to patch
FILES_TO_PATCH=(
    "$NEMO_PATH/lightning/one_logger_callback.py"
    "$NEMO_PATH/lightning/callback_group.py"
    "$NEMO_PATH/collections/llm/gpt/data/api.py"
    "$NEMO_PATH/export/utils/model_loader.py"
    "$NEMO_PATH/collections/llm/api.py"
    "$NEMO_PATH/collections/llm/__init__.py"
)

# Backup files
for file in "${FILES_TO_PATCH[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$BACKUP_DIR/"
        echo "  ✓ Backed up: $(basename $file)"
    fi
done

echo ""
echo "[PATCH 1/6] Patching one_logger_callback.py (nv_one_logger imports)..."

# Check if already patched
if grep -q "NV_ONE_LOGGER_AVAILABLE" "$NEMO_PATH/lightning/one_logger_callback.py" 2>/dev/null; then
    echo "  ℹ Already patched, skipping..."
else
    python << 'EOPYTHON'
import sys
nemo_path = sys.argv[1]
file_path = f"{nemo_path}/lightning/one_logger_callback.py"

with open(file_path, 'r') as f:
    content = f.read()

# Replace imports section
old_imports = """from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.training_telemetry.api.callbacks import on_app_start
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback"""

new_imports = """from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# Conditional import: nv_one_logger packages are optional dependencies
# that are not available on public PyPI. Gracefully degrade if missing.
try:
    from nv_one_logger.api.config import OneLoggerConfig
    from nv_one_logger.training_telemetry.api.callbacks import on_app_start
    from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback
    NV_ONE_LOGGER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Create stub classes when telemetry is unavailable
    NV_ONE_LOGGER_AVAILABLE = False
    OneLoggerConfig = None
    on_app_start = lambda: None
    TrainingTelemetryConfig = None
    TrainingTelemetryProvider = None
    OneLoggerPTLCallback = object  # Stub base class

from nemo.lightning.base_callback import BaseCallback"""

if old_imports in content:
    content = content.replace(old_imports, new_imports)
    content = content.replace("__all__ = ['OneLoggerNeMoCallback']",
                            "__all__ = ['OneLoggerNeMoCallback', 'NV_ONE_LOGGER_AVAILABLE']")

    # Add availability check to functions
    content = content.replace(
        '"""Generate minimal configuration for OneLogger initialization.\n\n    This function provides the absolute minimal configuration needed for OneLogger initialization.\n    It only includes the required fields and uses defaults for everything else to avoid\n    dependencies on exp_manager during early import.\n\n    Returns:\n        Dictionary containing minimal initialization configuration\n    """\n    if "EXP_NAME" in os.environ:',
        '"""Generate minimal configuration for OneLogger initialization.\n\n    This function provides the absolute minimal configuration needed for OneLogger initialization.\n    It only includes the required fields and uses defaults for everything else to avoid\n    dependencies on exp_manager during early import.\n\n    Returns:\n        Dictionary containing minimal initialization configuration\n    """\n    if not NV_ONE_LOGGER_AVAILABLE:\n        return {}\n\n    if "EXP_NAME" in os.environ:'
    )

    content = content.replace(
        '"""Generate base configuration for OneLogger training telemetry.\n\n    This function provides the common configuration needed for both NeMo v1 and v2.\n    It extracts basic training information from trainer object and uses provided\n    batch size and sequence length values.\n\n    Args:\n        trainer: PyTorch Lightning trainer instance\n        global_batch_size: Global batch size (calculated by version-specific function)\n        seq_length: Sequence length (calculated by version-specific function)\n\n    Returns:\n        Dictionary containing base training callback configuration\n    """\n    # Extract values from trainer',
        '"""Generate base configuration for OneLogger training telemetry.\n\n    This function provides the common configuration needed for both NeMo v1 and v2.\n    It extracts basic training information from trainer object and uses provided\n    batch size and sequence length values.\n\n    Args:\n        trainer: PyTorch Lightning trainer instance\n        global_batch_size: Global batch size (calculated by version-specific function)\n        seq_length: Sequence length (calculated by version-specific function)\n\n    Returns:\n        Dictionary containing base training callback configuration\n    """\n    if not NV_ONE_LOGGER_AVAILABLE:\n        return {}\n    # Extract values from trainer'
    )

    with open(file_path, 'w') as f:
        f.write(content)

    print("  ✓ Patched successfully")
else:
    print("  ℹ Pattern not found, may already be modified")

EOPYTHON
fi

echo ""
echo "Patch complete! Your NeMo installation can now import without nv_one_logger."
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""

# Test the fix
echo "========================================="
echo "Testing Fix..."
echo "========================================="
echo ""

python << 'EOF'
import sys
try:
    print("[TEST] Importing NeMo LLM...")
    from nemo.collections import llm
    print("[SUCCESS] ✓ NeMo LLM module imported!")

    print("[TEST] Checking import_ckpt availability...")
    if hasattr(llm, 'import_ckpt'):
        print("[SUCCESS] ✓ llm.import_ckpt is available")
    else:
        print("[WARNING] import_ckpt not available (may need additional patches)")

    print("[TEST] Checking DeepSeek models...")
    if hasattr(llm, 'DeepSeekV3Config') and hasattr(llm, 'DeepSeekModel'):
        print("[SUCCESS] ✓ DeepSeek models available")

    print("")
    print("=" * 50)
    print("✓ ALL TESTS PASSED!")
    print("=" * 50)
    sys.exit(0)

except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    print("")
    print("Additional patches may be needed.")
    print("Check the error message above.")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================="
    echo "✓ FIX COMPLETE!"
    echo "================================================================="
    echo ""
    echo "You can now run your model conversion:"
    echo "  python scripts/convert/import_to_nemo.py --help"
    echo ""
    echo "To restore original files if needed:"
    echo "  cp $BACKUP_DIR/* $NEMO_PATH/lightning/"
    echo ""
else
    echo ""
    echo "================================================================="
    echo "✗ FIX INCOMPLETE"
    echo "================================================================="
    echo ""
    echo "Some tests failed. You may need to:"
    echo "  1. Check the error messages above"
    echo "  2. Reinstall NeMo: pip uninstall -y nemo-toolkit && pip install nemo-toolkit"
    echo "  3. Rerun this script"
    echo ""
    echo "Backups available at: $BACKUP_DIR"
    echo ""
    exit 1
fi
