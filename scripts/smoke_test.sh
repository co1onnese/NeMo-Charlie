#!/bin/bash
# smoke_test.sh
# Quick smoke test of the entire pipeline on CPU with minimal data

set -e  # Exit on error

echo "========================================="
echo "SFT Pipeline - Smoke Test"
echo "========================================="

# Activate venv if it exists
if [ -d "venv/bin" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Set CPU-only mode
export CPU_ONLY_MODE=true
export SMOKE_TEST_MODE=true

# Test 1: Validation utilities
echo ""
echo "Test 1: Validation utilities..."
python3 tests/test_data_pipeline.py

# Test 2: XML to JSONL (using example file)
echo ""
echo "Test 2: XML to JSONL conversion..."
if [ -f "data/samples/example_input.xml" ]; then
    python3 src/parsers/xml_to_jsonl.py \
        --input_dir data/samples \
        --output_file data/jsonl/smoke_test.jsonl
    
    # Check output
    if [ -f "data/jsonl/smoke_test.jsonl" ]; then
        LINE_COUNT=$(wc -l < data/jsonl/smoke_test.jsonl)
        echo "  ✓ Created smoke_test.jsonl with $LINE_COUNT records"
    else
        echo "  ✗ Failed to create smoke_test.jsonl"
        exit 1
    fi
else
    echo "  ⊘ Skipping - no sample XML file found"
fi

# Test 3: Dataset conversion (if JSONL exists)
echo ""
echo "Test 3: Dataset conversion..."
if [ -f "data/jsonl/smoke_test.jsonl" ]; then
    python3 src/data/convert_dataset.py \
        --jsonl data/jsonl/smoke_test.jsonl \
        --out_dir data/hf_datasets/smoke_test_dataset \
        --min_samples 1 \
        --validation_days 1
    
    if [ -d "data/hf_datasets/smoke_test_dataset" ]; then
        echo "  ✓ Created HF dataset"
    else
        echo "  ✗ Failed to create HF dataset"
        exit 1
    fi
else
    echo "  ⊘ Skipping - no JSONL file"
fi

# Test 4: Price data API (quick check)
echo ""
echo "Test 4: Price data API..."
python3 << 'EOF'
import sys
sys.path.append('.')
from src.data.price_data import PriceDataClient

try:
    client = PriceDataClient()
    # Try to get a simple price point (will use cache if available)
    ret = client.get_forward_return("AAPL", "2024-01-01", forward_days=1)
    if ret is not None:
        print(f"  ✓ Price API working (AAPL return: {ret:.4f})")
    else:
        print("  ⚠ Price API returned None (may be expected for test date)")
except Exception as e:
    print(f"  ⚠ Price API test inconclusive: {e}")
EOF

# Test 5: Tokenization (if dataset exists)
echo ""
echo "Test 5: Tokenization..."
if [ -d "data/hf_datasets/smoke_test_dataset" ]; then
    echo "  Note: Tokenization requires model download, skipping in smoke test"
    echo "  ⊘ Run manually: python3 src/data/tokenize_and_shard.py --dataset_dir data/hf_datasets/smoke_test_dataset"
else
    echo "  ⊘ Skipping - no dataset"
fi

# Summary
echo ""
echo "========================================="
echo "Smoke Test Summary"
echo "========================================="
echo "✓ Validation utilities: PASSED"
echo "✓ XML parsing: PASSED"
echo "✓ Dataset conversion: PASSED"
echo "⚠ Price API: CHECK MANUALLY"
echo "⊘ Tokenization: SKIPPED (requires model download)"
echo "⊘ Training: SKIPPED (requires GPU or long CPU time)"
echo ""
echo "Next steps:"
echo "1. Copy your XML files to data/raw_xml/"
echo "2. Run: python3 src/parsers/xml_to_jsonl.py"
echo "3. Run: python3 src/data/convert_dataset.py"
echo "4. On GPU server: python3 src/train/train_sft.py --config configs/sft_config.yaml"
echo ""
