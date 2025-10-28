#!/bin/bash
# verify_nemo_fixes.sh
# Verifies that NeMo optional dependencies patches are applied and working

set -e

echo "================================================================="
echo "NeMo Optional Dependencies - Verification Script"
echo "================================================================="
echo ""

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[ERROR] Virtual environment not activated!"
    echo "[ERROR] Please run:"
    echo "[ERROR]   source venv/bin/activate"
    echo "[ERROR]   bash scripts/verify_nemo_fixes.sh"
    exit 1
fi

echo "[INFO] Virtual environment: $VIRTUAL_ENV"
echo ""

# Find NeMo installation
NEMO_PATH=$(python -c "import nemo; import os; print(os.path.dirname(nemo.__file__))" 2>/dev/null)
if [ -z "$NEMO_PATH" ]; then
    echo "[ERROR] NeMo not installed"
    exit 1
fi

echo "[INFO] NeMo installation: $NEMO_PATH"
echo ""
echo "========================================="
echo "Checking Patches..."
echo "========================================="
echo ""

# Check each patched file
check_patch() {
    local file=$1
    local marker=$2
    local description=$3

    if grep -q "$marker" "$file" 2>/dev/null; then
        echo "  ✓ $description"
        return 0
    else
        echo "  ✗ $description - MISSING"
        return 1
    fi
}

PATCHES_OK=true

check_patch "$NEMO_PATH/lightning/one_logger_callback.py" "NV_ONE_LOGGER_AVAILABLE" "one_logger_callback.py (telemetry)" || PATCHES_OK=false
check_patch "$NEMO_PATH/lightning/callback_group.py" "NV_ONE_LOGGER_AVAILABLE" "callback_group.py (callbacks)" || PATCHES_OK=false
check_patch "$NEMO_PATH/collections/llm/gpt/data/api.py" "NEMO_RUN_AVAILABLE" "gpt/data/api.py (data modules)" || PATCHES_OK=false
check_patch "$NEMO_PATH/export/utils/model_loader.py" "TENSORSTORE_AVAILABLE" "model_loader.py (export)" || PATCHES_OK=false
check_patch "$NEMO_PATH/collections/llm/api.py" "NEMO_RUN_AVAILABLE" "llm/api.py (API functions)" || PATCHES_OK=false

echo ""
echo "========================================="
echo "Testing NeMo Import..."
echo "========================================="
echo ""

# Test NeMo import
python << 'EOF'
import sys

try:
    print("[TEST] Importing NeMo LLM...")
    from nemo.collections import llm
    print("  ✓ NeMo LLM module imported successfully")

    print("[TEST] Checking import_ckpt...")
    if hasattr(llm, 'import_ckpt'):
        print("  ✓ llm.import_ckpt is available")
    else:
        print("  ✗ llm.import_ckpt NOT available")
        sys.exit(1)

    print("[TEST] Checking DeepSeek models...")
    if hasattr(llm, 'DeepSeekV3Config') and hasattr(llm, 'DeepSeekModel'):
        print("  ✓ DeepSeek models available")
    else:
        print("  ✗ DeepSeek models NOT available")
        sys.exit(1)

    sys.exit(0)

except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

TEST_RESULT=$?

echo ""
echo "================================================================="

if [ "$PATCHES_OK" = true ] && [ $TEST_RESULT -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED!"
    echo "================================================================="
    echo ""
    echo "Your NeMo installation is properly patched and working."
    echo "You can now run your model conversion:"
    echo ""
    echo "  python scripts/convert/import_to_nemo.py --help"
    echo ""
    exit 0
else
    echo "✗ SOME CHECKS FAILED"
    echo "================================================================="
    echo ""
    if [ "$PATCHES_OK" = false ]; then
        echo "Some patches are missing. You may need to:"
        echo "  1. Reinstall NeMo and reapply patches"
        echo "  2. Check NEMO_FIXES.md for manual patching instructions"
    fi
    if [ $TEST_RESULT -ne 0 ]; then
        echo "NeMo import tests failed."
        echo "Check the error messages above for details."
    fi
    echo ""
    exit 1
fi
