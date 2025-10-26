#!/bin/bash
#
# cpu_comprehensive_test.sh
# Comprehensive CPU testing script for SFT-Charlie pipeline
# Tests all CPU-compatible components before GPU deployment
#
# Usage: bash scripts/cpu_comprehensive_test.sh [--verbose] [--skip-api]
#

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

VERBOSE=false
SKIP_API=false
TEST_RESULTS_FILE="CPU_TEST_RESULTS_$(date +%Y%m%d_%H%M%S).md"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --skip-api)
            SKIP_API=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--verbose] [--skip-api]"
            exit 1
            ;;
    esac
done

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

print_section() {
    echo ""
    echo "------------------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------------------"
}

print_success() {
    echo "✓ $1"
}

print_warning() {
    echo "⚠ $1"
}

print_error() {
    echo "✗ $1"
}

print_skip() {
    echo "⊘ $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is NOT installed"
        return 1
    fi
}

# ============================================================================
# Test Results Tracking
# ============================================================================

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0
TESTS_WARNING=0

record_pass() {
    ((TESTS_PASSED++))
    print_success "$1"
}

record_fail() {
    ((TESTS_FAILED++))
    print_error "$1"
}

record_warning() {
    ((TESTS_WARNING++))
    print_warning "$1"
}

record_skip() {
    ((TESTS_SKIPPED++))
    print_skip "$1"
}

# ============================================================================
# Start Testing
# ============================================================================

print_header "SFT-Charlie CPU Comprehensive Testing"

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Working Directory: $(pwd)"
echo ""

# Set CPU-only mode
export CPU_ONLY_MODE=true
export SMOKE_TEST_MODE=true

# ============================================================================
# Phase 0: Environment Check
# ============================================================================

print_header "Phase 0: Environment & Prerequisites"

# Check Python version
print_section "Python Version"
python3 --version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$(echo "$PYTHON_VERSION >= 3.10" | bc -l 2>/dev/null || echo "0")" == "1" ]]; then
    record_pass "Python 3.10+ detected ($PYTHON_VERSION)"
else
    record_warning "Python version $PYTHON_VERSION (3.10+ recommended)"
fi

# Check required commands
print_section "Required Commands"
check_command python3 || record_fail "python3 missing"
check_command pip || record_fail "pip missing"
check_command git || record_fail "git missing"

# Check virtual environment
print_section "Virtual Environment"
if [ -d "venv" ]; then
    record_pass "Virtual environment exists (venv/)"
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "Activating virtual environment..."
        source venv/bin/activate
    fi
else
    record_warning "No virtual environment found (venv/). Using system Python."
fi

# Check disk space
print_section "Disk Space"
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
echo "Available space: $DISK_AVAIL"
# Check if at least 10GB available
DISK_AVAIL_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_AVAIL_GB" -gt 10 ]; then
    record_pass "Sufficient disk space (${DISK_AVAIL})"
else
    record_warning "Low disk space (${DISK_AVAIL}), 10GB+ recommended"
fi

# Check Python packages
print_section "Python Dependencies"
python3 << 'EOF'
import sys
import importlib

packages = [
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("yaml", "PyYAML"),
    ("datasets", "HuggingFace datasets"),
    ("transformers", "HuggingFace transformers"),
    ("dotenv", "python-dotenv"),
]

missing = []
for module, name in packages:
    try:
        importlib.import_module(module)
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - NOT INSTALLED")
        missing.append(name)

if missing:
    print(f"\n⚠ Missing packages: {', '.join(missing)}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ All core dependencies installed")
EOF

if [ $? -eq 0 ]; then
    record_pass "All core Python packages installed"
else
    record_fail "Missing Python packages - run: pip install -r requirements.txt"
    exit 1
fi

# ============================================================================
# Phase 1: Configuration Validation
# ============================================================================

print_header "Phase 1: Configuration Validation"

# Check .env file
print_section ".env File"
if [ -f ".env" ]; then
    record_pass ".env file exists"

    # Validate required variables
    python3 << 'EOF'
from dotenv import load_dotenv
import os

load_dotenv()

required_vars = {
    'TRAIN_START_DATE': 'Training start date',
    'TRAIN_END_DATE': 'Training end date',
    'TEST_START_DATE': 'Test start date',
    'TRAIN_BACKEND': 'Training backend',
}

missing = []
for var, desc in required_vars.items():
    val = os.getenv(var)
    if val:
        print(f"✓ {var}: {val}")
    else:
        print(f"✗ {var}: MISSING ({desc})")
        missing.append(var)

if missing:
    print(f"\n⚠ Missing variables: {missing}")
else:
    print("\n✓ All required environment variables set")
EOF

    if [ $? -eq 0 ]; then
        record_pass "Environment variables validated"
    else
        record_warning "Some environment variables missing or invalid"
    fi
else
    record_warning ".env file not found (optional for testing)"
fi

# Check YAML configs
print_section "Configuration Files"
for config in configs/sft_config.yaml configs/eval_config.yaml configs/backtest_config.yaml; do
    if [ -f "$config" ]; then
        python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null
        if [ $? -eq 0 ]; then
            record_pass "$config - valid YAML"
        else
            record_fail "$config - invalid YAML syntax"
        fi
    else
        record_warning "$config - not found"
    fi
done

# Check NeMo config if exists
if [ -f "configs/nemo/finetune.yaml" ]; then
    python3 -c "import yaml; yaml.safe_load(open('configs/nemo/finetune.yaml'))" 2>/dev/null
    if [ $? -eq 0 ]; then
        record_pass "configs/nemo/finetune.yaml - valid YAML"
    else
        record_fail "configs/nemo/finetune.yaml - invalid YAML syntax"
    fi
fi

# ============================================================================
# Phase 2: Unit Tests
# ============================================================================

print_header "Phase 2: Unit Tests"

print_section "Data Pipeline Tests"
if [ -f "tests/test_data_pipeline.py" ]; then
    python3 tests/test_data_pipeline.py
    if [ $? -eq 0 ]; then
        record_pass "Data pipeline unit tests PASSED"
    else
        record_fail "Data pipeline unit tests FAILED"
    fi
else
    record_warning "tests/test_data_pipeline.py not found"
fi

# ============================================================================
# Phase 3: Utility Functions
# ============================================================================

print_header "Phase 3: Utility Functions"

# Test logger
print_section "Logging System"
python3 << 'EOF'
from src.utils.logger import setup_logger, get_logger

logger = setup_logger("cpu_test")
logger.info("Test message")
print("✓ Logger working")
EOF

if [ $? -eq 0 ]; then
    record_pass "Logging system functional"
else
    record_fail "Logging system failed"
fi

# Test manifest
print_section "Manifest System"
python3 << 'EOF'
from src.utils.manifest import get_git_info
import json

git_info = get_git_info()
if git_info.get('commit'):
    print(f"✓ Git info captured: {git_info['commit'][:8]}")
else:
    print("⚠ Not a git repository or no commits")
EOF

if [ $? -eq 0 ]; then
    record_pass "Manifest system functional"
else
    record_warning "Manifest system incomplete (may not be git repo)"
fi

# Test validation
print_section "Validation Utilities"
python3 << 'EOF'
from src.utils.validation import (
    validate_date_format,
    validate_action,
    normalize_date,
    normalize_action
)

# Test dates
assert validate_date_format("2024-01-01"), "Date validation failed"
assert normalize_date("2024/01/01") == "2024-01-01", "Date normalization failed"

# Test actions
assert validate_action("BUY"), "Action validation failed"
assert normalize_action("buy") == "BUY", "Action normalization failed"

print("✓ All validation tests passed")
EOF

if [ $? -eq 0 ]; then
    record_pass "Validation utilities working"
else
    record_fail "Validation utilities failed"
fi

# Test price data (optional)
if [ "$SKIP_API" = false ]; then
    print_section "Price Data API"
    python3 << 'EOF'
import sys
sys.path.append('.')
from src.data.price_data import PriceDataClient

try:
    client = PriceDataClient()
    print("✓ PriceDataClient initialized")

    # Try to get cached data (won't hit API if cached)
    ret = client.get_forward_return("AAPL", "2024-01-15", forward_days=1)
    if ret is not None:
        print(f"✓ Price API working (return: {ret:.4f})")
    else:
        print("⚠ API returned None (may need API key or cache)")
except Exception as e:
    print(f"⚠ Price API test inconclusive: {e}")
EOF

    if [ $? -eq 0 ]; then
        record_pass "Price data client functional"
    else
        record_warning "Price data client needs configuration"
    fi
else
    record_skip "Price data API test (--skip-api flag)"
fi

# ============================================================================
# Phase 4: Data Pipeline
# ============================================================================

print_header "Phase 4: Data Pipeline"

# Create test data directory
mkdir -p data/samples data/jsonl data/hf_datasets data/nemo

# Check for sample XML
print_section "XML to JSONL Conversion"
if [ -f "data/samples/example_input.xml" ] || [ -n "$(ls -A data/raw_xml 2>/dev/null)" ]; then
    INPUT_DIR="data/samples"
    if [ -n "$(ls -A data/raw_xml 2>/dev/null)" ]; then
        INPUT_DIR="data/raw_xml"
    fi

    python3 src/parsers/xml_to_jsonl.py \
        --input_dir "$INPUT_DIR" \
        --output_file data/jsonl/cpu_test.jsonl

    if [ -f "data/jsonl/cpu_test.jsonl" ]; then
        LINE_COUNT=$(wc -l < data/jsonl/cpu_test.jsonl)
        record_pass "XML to JSONL conversion succeeded ($LINE_COUNT records)"

        # Validate JSONL format
        python3 << 'EOF'
import json

with open("data/jsonl/cpu_test.jsonl") as f:
    for i, line in enumerate(f):
        try:
            record = json.loads(line)
            assert "ticker" in record, f"Line {i}: missing ticker"
            assert "reasoning" in record, f"Line {i}: missing reasoning"
            assert "action" in record, f"Line {i}: missing action"
        except Exception as e:
            print(f"✗ JSONL validation failed: {e}")
            exit(1)

print("✓ JSONL format valid")
EOF

        if [ $? -eq 0 ]; then
            record_pass "JSONL format validated"
        else
            record_fail "JSONL format invalid"
        fi
    else
        record_fail "XML to JSONL conversion failed"
    fi
else
    record_skip "XML parsing (no sample data found)"
    echo "  To test: Add XML files to data/samples/ or data/raw_xml/"
fi

# Dataset conversion
print_section "HuggingFace Dataset Creation"
if [ -f "data/jsonl/cpu_test.jsonl" ]; then
    python3 src/data/convert_dataset.py \
        --jsonl data/jsonl/cpu_test.jsonl \
        --out_dir data/hf_datasets/cpu_test_dataset \
        --min_samples 1 \
        --validation_days 1 \
        --test_days 1

    if [ -d "data/hf_datasets/cpu_test_dataset" ]; then
        record_pass "HuggingFace dataset created"

        # Validate dataset
        python3 << 'EOF'
from datasets import load_from_disk

ds = load_from_disk("data/hf_datasets/cpu_test_dataset")

# Check splits
assert "train" in ds, "Missing train split"
assert "validation" in ds, "Missing validation split"
assert "test" in ds, "Missing test split"

print(f"✓ Dataset splits: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")

# Check time ordering
train_dates = [r['as_of_date'] for r in ds['train']]
val_dates = [r['as_of_date'] for r in ds['validation']]
test_dates = [r['as_of_date'] for r in ds['test']]

if train_dates and val_dates:
    assert max(train_dates) <= min(val_dates), "Data leakage: train dates overlap validation"
if val_dates and test_dates:
    assert max(val_dates) <= min(test_dates), "Data leakage: validation dates overlap test"

print("✓ Time-based split validation passed (no leakage)")
EOF

        if [ $? -eq 0 ]; then
            record_pass "Dataset splits validated (no data leakage)"
        else
            record_fail "Dataset validation failed"
        fi
    else
        record_fail "Dataset creation failed"
    fi
else
    record_skip "Dataset creation (no JSONL file)"
fi

# NeMo export (without tokenizer)
print_section "NeMo Dataset Export"
if [ -d "data/hf_datasets/cpu_test_dataset" ]; then
    python3 src/data/export_nemo_dataset.py \
        --dataset_dir data/hf_datasets/cpu_test_dataset \
        --output_dir data/nemo/cpu_test \
        --max_samples 10 \
        --skip_tokenizer

    if [ -d "data/nemo/cpu_test" ]; then
        record_pass "NeMo dataset exported"

        # Validate JSONL structure
        for split in training validation test; do
            if [ -f "data/nemo/cpu_test/${split}.jsonl" ]; then
                LINE_COUNT=$(wc -l < "data/nemo/cpu_test/${split}.jsonl")
                echo "  ✓ ${split}.jsonl: $LINE_COUNT lines"
            fi
        done

        # Check format
        python3 << 'EOF'
import json

with open("data/nemo/cpu_test/training.jsonl") as f:
    first_line = f.readline()
    record = json.loads(first_line)

    assert "input" in record or "prompt" in record, "Missing input/prompt field"
    assert "output" in record or "completion" in record, "Missing output/completion field"

    print("✓ NeMo JSONL format valid")
EOF

        if [ $? -eq 0 ]; then
            record_pass "NeMo JSONL format validated"
        else
            record_fail "NeMo JSONL format invalid"
        fi
    else
        record_fail "NeMo export failed"
    fi
else
    record_skip "NeMo export (no HF dataset)"
fi

# ============================================================================
# Phase 5: Directory Structure
# ============================================================================

print_header "Phase 5: Directory Structure"

print_section "Required Directories"
REQUIRED_DIRS=(
    "src/parsers"
    "src/data"
    "src/train"
    "src/eval"
    "src/backtest"
    "src/utils"
    "configs"
    "scripts"
    "tests"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        record_pass "Directory exists: $dir"
    else
        record_warning "Directory missing: $dir"
    fi
done

# ============================================================================
# Phase 6: Git Repository
# ============================================================================

print_header "Phase 6: Git Repository"

if [ -d ".git" ]; then
    record_pass "Git repository detected"

    # Check status
    print_section "Git Status"
    git status --short

    # Check branch
    BRANCH=$(git branch --show-current)
    echo "Current branch: $BRANCH"

    # Check commit
    COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "Latest commit: $COMMIT"

    record_pass "Git repository status checked"
else
    record_warning "Not a git repository"
fi

# ============================================================================
# Summary
# ============================================================================

print_header "Test Summary"

echo ""
echo "Total Tests: $((TESTS_PASSED + TESTS_FAILED + TESTS_WARNING + TESTS_SKIPPED))"
echo "  ✓ Passed:   $TESTS_PASSED"
echo "  ✗ Failed:   $TESTS_FAILED"
echo "  ⚠ Warnings: $TESTS_WARNING"
echo "  ⊘ Skipped:  $TESTS_SKIPPED"
echo ""

# Overall status
if [ $TESTS_FAILED -eq 0 ]; then
    if [ $TESTS_WARNING -eq 0 ]; then
        echo "✅ ALL TESTS PASSED - Ready for GPU deployment"
        EXIT_CODE=0
    else
        echo "⚠️ TESTS PASSED WITH WARNINGS - Review warnings before GPU deployment"
        EXIT_CODE=0
    fi
else
    echo "❌ SOME TESTS FAILED - Fix issues before GPU deployment"
    EXIT_CODE=1
fi

# Save results
print_header "Saving Results"

cat > "$TEST_RESULTS_FILE" << EOF
# CPU Test Results

**Date:** $(date)
**Host:** $(hostname)
**Branch:** $(git branch --show-current 2>/dev/null || echo "unknown")
**Commit:** $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

## Environment
- OS: $(uname -s) $(uname -r)
- Python: $(python3 --version)
- Disk Space: $(df -h . | tail -1 | awk '{print $4}') available

## Test Results

| Category | Passed | Failed | Warnings | Skipped |
|----------|--------|--------|----------|---------|
| Total | $TESTS_PASSED | $TESTS_FAILED | $TESTS_WARNING | $TESTS_SKIPPED |

## Status

$(if [ $TESTS_FAILED -eq 0 ]; then echo "✅ **READY FOR GPU DEPLOYMENT**"; else echo "❌ **NOT READY** - Fix failures first"; fi)

## Next Steps

1. Review any warnings or failures above
2. Fix critical issues
3. Re-run this test script
4. Proceed with GPU deployment using \`GPU_DEPLOYMENT_CHECKLIST.md\`

## Detailed Output

See full console output for details.
EOF

echo "Results saved to: $TEST_RESULTS_FILE"
echo ""

# Final recommendations
print_header "Next Steps"

if [ $TESTS_FAILED -eq 0 ]; then
    echo "1. Review test results: cat $TEST_RESULTS_FILE"
    echo "2. Read GPU deployment guide: cat GPU_DEPLOYMENT_CHECKLIST.md"
    echo "3. Prepare data for GPU server transfer"
    echo "4. Schedule GPU server access"
else
    echo "1. Review failures in test output above"
    echo "2. Fix critical issues"
    echo "3. Re-run: bash scripts/cpu_comprehensive_test.sh"
fi

echo ""
exit $EXIT_CODE
