#!/bin/bash
# fix_modelopt.sh
# Fixes broken modelopt installation that prevents NeMo LLM import

set -e

echo "========================================="
echo "Fixing Broken Modelopt Installation"
echo "========================================="
echo ""

# Check if venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[ERROR] Virtual environment not activated!"
    echo "[ERROR] Please run:"
    echo "[ERROR]   source venv/bin/activate"
    echo "[ERROR]   bash scripts/fix_modelopt.sh"
    exit 1
fi

echo "[INFO] Virtual environment: $VIRTUAL_ENV"
echo ""

# Check if modelopt is installed
if pip show modelopt &>/dev/null || pip show nvidia-modelopt &>/dev/null; then
    echo "[INFO] Found modelopt installation. Removing..."
    pip uninstall -y modelopt nvidia-modelopt 2>/dev/null || true
    echo "✓ Removed broken modelopt"
else
    echo "[INFO] modelopt not found in pip (but may exist as broken files)"
fi

# Try to clean up any remaining modelopt files
MODELOPT_DIR="$VIRTUAL_ENV/lib/python*/site-packages/modelopt"
if ls $MODELOPT_DIR 2>/dev/null; then
    echo "[INFO] Removing modelopt directory..."
    rm -rf $MODELOPT_DIR
    echo "✓ Cleaned up modelopt directory"
fi

echo ""
echo "[INFO] Testing NeMo import..."

# Test if NeMo LLM works now
python3 << 'EOF'
import sys
try:
    from nemo.collections import llm
    if hasattr(llm, 'import_ckpt'):
        print("✓ NeMo LLM module works!")
        print("✓ llm.import_ckpt is available")
        sys.exit(0)
    else:
        print("✗ NeMo LLM imported but import_ckpt not available")
        print("This may be a NeMo version issue")
        sys.exit(1)
except Exception as e:
    print(f"✗ NeMo LLM import still failing: {e}")
    print("")
    print("You may need to reinstall NeMo:")
    print("  pip uninstall -y nemo-toolkit")
    print("  pip install nemo-toolkit[all]")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Fix Complete!"
    echo "========================================="
    echo ""
    echo "You can now run the model import:"
    echo "  bash scripts/run_full_pipeline.sh"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ Fix Failed"
    echo "========================================="
    echo ""
    echo "Additional troubleshooting needed."
    echo "Check the error messages above."
    echo ""
fi
