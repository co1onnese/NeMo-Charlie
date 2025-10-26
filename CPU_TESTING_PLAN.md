# CPU Testing Plan - SFT-Charlie Pipeline

**Purpose:** Validate all CPU-compatible components before deploying to GPU server.

**Server Context:** Current server has NO GPU. This plan tests everything possible without GPU hardware.

---

## Executive Summary

This testing plan validates ~80% of the pipeline functionality without requiring GPU access. It ensures data processing, validation, configuration, and setup scripts work correctly before incurring GPU costs.

**Testing Categories:**
- ✅ **Can Test on CPU** (Primary focus)
- ⚠️ **Limited Testing on CPU** (Reduced functionality)
- ❌ **Cannot Test on CPU** (Requires GPU)

---

## ✅ Phase 1: Core Data Pipeline (CPU-Compatible)

### 1.1 Validation Utilities
**Location:** `src/utils/validation.py`

**Tests:**
```bash
# Run existing unit tests
python3 tests/test_data_pipeline.py
```

**Manual Validation:**
- Date format validation (ISO 8601)
- Trading action validation (BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL)
- Thesis record completeness checks
- Time split validation (no data leakage)

**Expected Output:** All assertion tests pass

---

### 1.2 XML to JSONL Conversion
**Location:** `src/parsers/xml_to_jsonl.py`

**Prerequisites:**
- XML sample files exist in `data/samples/` or `data/raw_xml/`
- Python dependencies installed

**Test Command:**
```bash
# Test with sample data
python3 src/parsers/xml_to_jsonl.py \
    --input_dir data/samples \
    --output_file data/jsonl/cpu_test.jsonl

# Validate output
cat data/jsonl/cpu_test.jsonl | python3 -m json.tool | head -50
```

**Validation Checks:**
- [ ] JSONL file created successfully
- [ ] Each line is valid JSON
- [ ] Required fields present: `ticker`, `as_of_date`, `reasoning`, `action`
- [ ] Dates in ISO format (YYYY-MM-DD)
- [ ] Actions are valid (BUY/SELL/HOLD/STRONG_BUY/STRONG_SELL)
- [ ] No XML parsing errors in logs

**Expected Output:**
- `.jsonl` file with one JSON object per line
- Log messages showing parsed records count
- No error messages

---

### 1.3 HuggingFace Dataset Creation
**Location:** `src/data/convert_dataset.py`

**Prerequisites:**
- JSONL file from step 1.2
- Sufficient records for train/val/test split

**Test Command:**
```bash
# Create time-based splits
python3 src/data/convert_dataset.py \
    --jsonl data/jsonl/cpu_test.jsonl \
    --out_dir data/hf_datasets/cpu_test_dataset \
    --min_samples 1 \
    --validation_days 30 \
    --test_days 30

# Inspect dataset
python3 << 'EOF'
from datasets import load_from_disk
ds = load_from_disk("data/hf_datasets/cpu_test_dataset")
print(f"Train: {len(ds['train'])} samples")
print(f"Validation: {len(ds['validation'])} samples")
print(f"Test: {len(ds['test'])} samples")
print("\nSample record:")
print(ds['train'][0])
EOF
```

**Validation Checks:**
- [ ] Dataset directory created with Arrow files
- [ ] Three splits exist: `train`, `validation`, `test`
- [ ] Time ordering: train dates < validation dates < test dates
- [ ] No data leakage between splits
- [ ] Required columns: `instruction`, `input`, `output`, `ticker`, `as_of_date`

**Expected Output:**
- HuggingFace DatasetDict with 3 splits
- Chronological ordering maintained
- Sample counts match expectations

---

### 1.4 NeMo Dataset Export (LIMITED)
**Location:** `src/data/export_nemo_dataset.py`

**Test Command:**
```bash
# Export small sample WITHOUT tokenizer loading
python3 src/data/export_nemo_dataset.py \
    --dataset_dir data/hf_datasets/cpu_test_dataset \
    --output_dir data/nemo/cpu_test \
    --max_samples 10 \
    --skip_tokenizer

# Validate JSONL structure
head -3 data/nemo/cpu_test/training.jsonl | python3 -m json.tool
```

**Validation Checks:**
- [ ] Three JSONL files created: `training.jsonl`, `validation.jsonl`, `test.jsonl`
- [ ] Each line has `input` and `output` fields
- [ ] Prompt template applied correctly
- [ ] Special tokens present: `<reasoning>`, `<support>`, `<action>`

**⚠️ Limitation:** Cannot test tokenizer extension without downloading base model

---

## ✅ Phase 2: Configuration & Environment

### 2.1 Environment Variables
**Location:** `.env`

**Test Command:**
```bash
# Validate .env file
python3 << 'EOF'
from dotenv import load_dotenv
import os

load_dotenv()

required_vars = [
    'EODHD_API_KEY',
    'TRAIN_START_DATE',
    'TRAIN_END_DATE',
    'TEST_START_DATE',
    'TRAIN_BACKEND'
]

missing = []
for var in required_vars:
    val = os.getenv(var)
    if val:
        print(f"✓ {var}: {val}")
    else:
        print(f"✗ {var}: MISSING")
        missing.append(var)

if missing:
    print(f"\n❌ Missing variables: {missing}")
    exit(1)
else:
    print("\n✅ All required variables present")
EOF
```

**Validation Checks:**
- [ ] `.env` file exists
- [ ] All required variables set
- [ ] Date formats valid (YYYY-MM-DD)
- [ ] API key present (even if placeholder)
- [ ] `TRAIN_BACKEND=nemo`

---

### 2.2 Configuration Files
**Locations:** `configs/*.yaml`, `configs/nemo/*.yaml`

**Test Commands:**
```bash
# Validate YAML syntax
for config in configs/*.yaml configs/nemo/*.yaml; do
    echo "Checking $config..."
    python3 -c "import yaml; yaml.safe_load(open('$config'))" && echo "✓" || echo "✗ INVALID"
done

# Inspect key configs
python3 << 'EOF'
import yaml
from pathlib import Path

configs = [
    'configs/sft_config.yaml',
    'configs/eval_config.yaml',
    'configs/backtest_config.yaml',
    'configs/nemo/finetune.yaml'
]

for config_path in configs:
    if Path(config_path).exists():
        print(f"\n{'='*60}")
        print(f"Config: {config_path}")
        print('='*60)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            print(yaml.dump(cfg, default_flow_style=False, indent=2))
    else:
        print(f"⚠️ Missing: {config_path}")
EOF
```

**Validation Checks:**
- [ ] All YAML files parse without errors
- [ ] NeMo config has required fields: `recipe`, `train`, `data`
- [ ] Paths in configs exist or are placeholder-ready
- [ ] Hyperparameters are reasonable values

---

### 2.3 Dependencies & Environment Setup
**Location:** `scripts/setup_env.sh`, `requirements*.txt`

**Test Commands:**
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Validate requirements files exist
ls -lh requirements.txt requirements_nemo.txt

# Test environment setup (dry run)
export DRY_RUN=true
bash scripts/setup_env.sh

# Check installed packages
pip list | grep -E "torch|transformers|datasets|pandas|numpy|pyyaml"
```

**Validation Checks:**
- [ ] Python 3.10+ installed
- [ ] Requirements files exist and parseable
- [ ] Setup script runs without errors (dry run mode)
- [ ] Core packages installed (without GPU versions)

---

## ✅ Phase 3: Utilities & Support Functions

### 3.1 Price Data API
**Location:** `src/data/price_data.py`

**Test Command:**
```bash
python3 << 'EOF'
import sys
sys.path.append('.')
from src.data.price_data import PriceDataClient
from dotenv import load_dotenv

load_dotenv()

try:
    client = PriceDataClient()
    print("✓ PriceDataClient initialized")

    # Test with well-known ticker (will use cache if available)
    ret = client.get_forward_return("AAPL", "2024-01-15", forward_days=5)

    if ret is not None:
        print(f"✓ Price API working: AAPL 5-day return from 2024-01-15: {ret:.4f}")
    else:
        print("⚠️ API returned None (may be cache miss or API limit)")

    # Check cache
    cache_dir = "data/price_cache"
    import os
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        print(f"✓ Cache directory exists with {len(files)} files")

except Exception as e:
    print(f"✗ Price API test failed: {e}")
    import traceback
    traceback.print_exc()
EOF
```

**Validation Checks:**
- [ ] Client initializes without errors
- [ ] API key loaded from environment
- [ ] Cache directory created
- [ ] Can retrieve price data (or graceful failure)

**⚠️ Note:** May hit API rate limits or require valid API key

---

### 3.2 Logging System
**Location:** `src/utils/logger.py`

**Test Command:**
```bash
python3 << 'EOF'
from src.utils.logger import setup_logger, get_logger

# Test logger setup
logger = setup_logger("test_logger")
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")

print("\n✓ Logger working - check logs/ directory for output")

# Check log directory
import os
if os.path.exists("logs"):
    log_files = os.listdir("logs")
    print(f"✓ Found {len(log_files)} log files")
else:
    print("⚠️ logs/ directory not created")
EOF
```

**Validation Checks:**
- [ ] Logger initializes without errors
- [ ] Messages appear in console with colors
- [ ] Log files created in `logs/` directory
- [ ] Different log levels work correctly

---

### 3.3 Manifest System
**Location:** `src/utils/manifest.py`

**Test Command:**
```bash
python3 << 'EOF'
from src.utils.manifest import create_manifest, get_git_info
import json

# Test git info
git_info = get_git_info()
print("Git Info:")
print(json.dumps(git_info, indent=2))

# Test manifest creation
manifest = create_manifest(
    config_path="configs/sft_config.yaml",
    data_files=["data/jsonl/cpu_test.jsonl"],
    run_name="cpu_test"
)

print("\nManifest:")
print(json.dumps(manifest, indent=2, default=str))

# Save to file
import os
os.makedirs("manifests", exist_ok=True)
with open("manifests/cpu_test.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print("\n✓ Manifest saved to manifests/cpu_test.json")
EOF
```

**Validation Checks:**
- [ ] Git info captured correctly
- [ ] Manifest includes all required fields
- [ ] Config hash computed
- [ ] Environment info captured
- [ ] Manifest file saved successfully

---

## ✅ Phase 4: Smoke Tests & Integration

### 4.1 Run Existing Smoke Test
**Location:** `scripts/smoke_test.sh`

**Test Command:**
```bash
# Ensure test data exists
mkdir -p data/samples

# Run smoke test
bash scripts/smoke_test.sh
```

**Expected Output:**
- ✓ Validation utilities: PASSED
- ✓ XML parsing: PASSED
- ✓ Dataset conversion: PASSED
- ⚠ Price API: CHECK MANUALLY
- ⊘ Tokenization: SKIPPED (requires model download)

---

### 4.2 Full Data Pipeline Test
**Test Command:**
```bash
#!/bin/bash
# Full CPU-compatible pipeline test

set -e

echo "========================================="
echo "Full CPU Pipeline Test"
echo "========================================="

# 1. XML to JSONL
echo -e "\n[1/5] XML to JSONL conversion..."
python3 src/parsers/xml_to_jsonl.py \
    --input_dir data/raw_xml \
    --output_file data/jsonl/full_test.jsonl

# 2. Dataset creation
echo -e "\n[2/5] Creating HF dataset..."
python3 src/data/convert_dataset.py \
    --jsonl data/jsonl/full_test.jsonl \
    --out_dir data/hf_datasets/full_test_dataset

# 3. NeMo export
echo -e "\n[3/5] Exporting to NeMo format..."
python3 src/data/export_nemo_dataset.py \
    --dataset_dir data/hf_datasets/full_test_dataset \
    --output_dir data/nemo/full_test \
    --max_samples 100 \
    --skip_tokenizer

# 4. Validation
echo -e "\n[4/5] Validating outputs..."
python3 tests/test_data_pipeline.py

# 5. Summary
echo -e "\n[5/5] Pipeline Summary..."
python3 << 'EOF'
import json
from pathlib import Path

# Check JSONL
jsonl_path = Path("data/jsonl/full_test.jsonl")
if jsonl_path.exists():
    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]
    print(f"✓ JSONL: {len(records)} records")

# Check HF dataset
from datasets import load_from_disk
ds_path = "data/hf_datasets/full_test_dataset"
if Path(ds_path).exists():
    ds = load_from_disk(ds_path)
    print(f"✓ HF Dataset: {len(ds['train'])} train, {len(ds['validation'])} val, {len(ds['test'])} test")

# Check NeMo files
nemo_path = Path("data/nemo/full_test")
if nemo_path.exists():
    files = list(nemo_path.glob("*.jsonl"))
    print(f"✓ NeMo JSONL: {len(files)} files")
    for f in files:
        with open(f) as fh:
            lines = sum(1 for _ in fh)
        print(f"  - {f.name}: {lines} lines")

print("\n✅ Full pipeline test complete!")
EOF

echo "========================================="
```

Save as `scripts/cpu_full_test.sh` and run:
```bash
bash scripts/cpu_full_test.sh
```

---

## ⚠️ Phase 5: Limited Testing (Partial CPU Support)

### 5.1 Tokenizer Testing (Small Model)
**Limitation:** Cannot test with DeepSeek-V3 (685B parameters) on CPU

**Alternative Test:**
```bash
# Test with tiny model instead
python3 << 'EOF'
from transformers import AutoTokenizer

# Use small model for testing tokenizer logic
model_name = "gpt2"  # 117M params - CPU compatible
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add special tokens
special_tokens = {
    "additional_special_tokens": [
        "<reasoning>", "</reasoning>",
        "<support>", "</support>",
        "<action>", "</action>",
        "<STRONG_BUY>", "<BUY>", "<HOLD>", "<SELL>", "<STRONG_SELL>"
    ]
}

num_added = tokenizer.add_special_tokens(special_tokens)
print(f"✓ Added {num_added} special tokens")
print(f"✓ Vocab size: {len(tokenizer)}")

# Test tokenization
sample = "Buy <reasoning>Strong fundamentals</reasoning> <action>BUY</action>"
tokens = tokenizer(sample)
print(f"✓ Tokenized sample: {len(tokens['input_ids'])} tokens")

print("\n⚠️ Note: This tests tokenizer LOGIC only, not DeepSeek-V3 compatibility")
EOF
```

---

### 5.2 Evaluation Logic (Without Model Inference)
**Test Command:**
```bash
# Test evaluation utilities without actual model
python3 << 'EOF'
from src.eval.metrics import compute_metrics
from src.utils.eval_utils import extract_action
import pandas as pd

# Test action extraction
test_outputs = [
    "Analysis shows growth. <action>BUY</action>",
    "Declining revenue. <action>SELL</action>",
    "No clear signal. <action>HOLD</action>"
]

print("Testing action extraction:")
for output in test_outputs:
    action = extract_action(output)
    print(f"  '{output[:30]}...' -> {action}")

# Test metrics computation (with dummy data)
predictions = ["BUY", "SELL", "HOLD", "BUY", "SELL"]
references = ["BUY", "SELL", "SELL", "BUY", "HOLD"]

metrics = compute_metrics(predictions, references)
print("\n✓ Metrics computation:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")

print("\n⚠️ Note: Testing logic only, not actual model predictions")
EOF
```

---

### 5.3 Backtesting Logic (Without Real Predictions)
**Test Command:**
```bash
# Test backtest with synthetic data
python3 << 'EOF'
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create synthetic evaluation data
dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

np.random.seed(42)
records = []
for date in dates[:100]:  # Use 100 days
    ticker = np.random.choice(tickers)
    records.append({
        "ticker": ticker,
        "as_of_date": date.strftime("%Y-%m-%d"),
        "predicted_action": np.random.choice(["BUY", "SELL", "HOLD"]),
        "true_action": np.random.choice(["BUY", "SELL", "HOLD"]),
        "forward_return_5d": np.random.randn() * 0.05  # 5% daily vol
    })

df = pd.DataFrame(records)
df.to_csv("results/synthetic_eval.csv", index=False)
print(f"✓ Created synthetic evaluation data: {len(df)} records")

# Test backtest import (without running)
try:
    from src.backtest.trading_backtest import run_backtest
    print("✓ Backtest module imports successfully")
    print("⚠️ Note: Not running actual backtest - requires real price data")
except Exception as e:
    print(f"✗ Backtest import failed: {e}")
EOF
```

---

## ❌ Phase 6: Cannot Test on CPU (GPU Required)

### 6.1 Model Conversion (FP8 → BF16 → NeMo)
**Why GPU Required:**
- FP8 → BF16 conversion uses Triton kernels (GPU-only)
- Model size (685B params) exceeds typical CPU RAM
- NeMo import requires CUDA-enabled PyTorch

**Files:** `scripts/convert/fp8_cast_bf16.py`, `scripts/convert/import_to_nemo.py`

**GPU Server Action Required:**
```bash
# Step 1: Download model (can do on CPU server if bandwidth available)
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base checkpoints/source/deepseek-v3

# Steps 2-3: Must run on GPU server
bash scripts/convert/convert_deepseek_v3.sh --source checkpoints/source/deepseek-v3 --output checkpoints/bf16/deepseek-v3
python3 scripts/convert/import_to_nemo.py --bf16-dir checkpoints/bf16/deepseek-v3 --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo
```

---

### 6.2 NeMo Training
**Why GPU Required:**
- Model requires 8×H100 80GB GPUs
- Tensor parallelism across 8 GPUs
- CUDA operations throughout training loop

**File:** `src/train/train_nemo.py`

**GPU Server Action Required:**
```bash
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/main
```

---

### 6.3 Model Evaluation (Full Scale)
**Why GPU Required:**
- Model inference requires GPU
- Batch processing of test set
- Model loaded in distributed manner

**File:** `src/eval/evaluate_nemo.py`

**Partial CPU Test:**
- Can test data loading and preprocessing
- Cannot test actual model inference

**GPU Server Action Required:**
```bash
python3 src/eval/evaluate_nemo.py \
    --model checkpoints/nemo_runs/main/final.nemo \
    --dataset data/nemo/sft_dataset \
    --results results/eval_results.csv
```

---

## 📊 Testing Progress Checklist

### Pre-Deployment Validation (CPU Server)

#### Core Pipeline
- [ ] Unit tests pass (`tests/test_data_pipeline.py`)
- [ ] XML parsing works with sample data
- [ ] JSONL conversion produces valid output
- [ ] HuggingFace dataset creation succeeds
- [ ] Time-based splits validated (no leakage)
- [ ] NeMo JSONL export works (without tokenizer)

#### Configuration
- [ ] `.env` file has all required variables
- [ ] YAML configs parse without errors
- [ ] NeMo config structure valid
- [ ] Paths in configs are correct

#### Utilities
- [ ] Logging system functional
- [ ] Manifest creation works
- [ ] Validation functions pass tests
- [ ] Price API initializes (with/without real key)

#### Integration
- [ ] Smoke test passes
- [ ] Full pipeline test succeeds
- [ ] All directories created correctly
- [ ] No import errors in any module

#### Documentation
- [ ] GPU requirements documented
- [ ] Setup scripts reviewed
- [ ] Known limitations documented
- [ ] Transfer checklist prepared

---

## 🚀 GPU Server Preparation Checklist

### Before Transfer
- [ ] All CPU tests passing
- [ ] Data files prepared and validated
- [ ] Configurations finalized
- [ ] `.env` file ready (with real API keys)
- [ ] Git repository clean (no uncommitted changes)

### On GPU Server
- [ ] CUDA 12.1+ installed
- [ ] NVIDIA drivers updated
- [ ] Python 3.10+ available
- [ ] Sufficient disk space (500GB+)
- [ ] Network access to HuggingFace
- [ ] Git LFS installed (for model download)

### Installation Steps
```bash
# 1. Clone repository
git clone <repo-url>
cd SFT-Charlie

# 2. Setup environment with GPU support
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
source venv/bin/activate

# 3. Verify CUDA
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

# 4. Transfer or regenerate data
rsync -avz --progress cpu-server:/path/to/data/ ./data/

# 5. Run GPU smoke test
python3 -c "import nemo; import megatron; print('NeMo OK')"
```

### First GPU Test
```bash
# Quick test with minimal data
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --smoke-test \
    --max_steps 10
```

---

## 📝 Test Execution Log Template

Save results as `CPU_TEST_RESULTS.md`:

```markdown
# CPU Test Results

**Date:** YYYY-MM-DD
**Tester:** [Name]
**Branch:** [git branch]
**Commit:** [git commit hash]

## Environment
- OS: [uname -a]
- Python: [python3 --version]
- Disk Space: [df -h]

## Test Results

### Phase 1: Data Pipeline
- [ ] test_data_pipeline.py: PASS/FAIL
- [ ] XML parsing: PASS/FAIL
- [ ] Dataset creation: PASS/FAIL
- [ ] NeMo export: PASS/FAIL

### Phase 2: Configuration
- [ ] .env validation: PASS/FAIL
- [ ] YAML configs: PASS/FAIL
- [ ] Dependencies: PASS/FAIL

### Phase 3: Utilities
- [ ] Price API: PASS/FAIL
- [ ] Logging: PASS/FAIL
- [ ] Manifest: PASS/FAIL

### Phase 4: Integration
- [ ] Smoke test: PASS/FAIL
- [ ] Full pipeline: PASS/FAIL

## Issues Found
[List any issues, with details]

## Ready for GPU Deployment?
YES / NO

[If NO, what needs to be fixed?]
```

---

## 🔍 Troubleshooting Common Issues

### "ModuleNotFoundError: No module named 'nemo'"
**Expected on CPU server** - NeMo requires CUDA. Skip NeMo imports in CPU tests.

**Solution:** Use `--skip_tokenizer` flag and skip training tests.

---

### "EODHD_API_KEY not found"
**Can proceed with placeholder** - Price API will fail gracefully.

**Solution:** Use dummy key for testing, replace on GPU server.

---

### "Insufficient samples for split"
**Need more data** - Minimum samples required for train/val/test.

**Solution:** Use `--min_samples 1` for testing, increase for production.

---

### "Disk space full"
**Check data directories** - Remove old test outputs.

**Solution:**
```bash
du -sh data/* checkpoints/* logs/*
rm -rf data/hf_datasets/test_*
```

---

## 📞 Next Steps

1. **Run CPU tests** following this plan
2. **Document results** using test log template
3. **Fix any issues** found during testing
4. **Prepare data transfer** to GPU server
5. **Review GPU checklist** before deployment
6. **Schedule GPU testing** once CPU validation complete

---

## Summary: What We Can/Cannot Test

| Component | CPU Test | GPU Test | Notes |
|-----------|----------|----------|-------|
| XML Parsing | ✅ Full | ✅ Full | Pure Python, no GPU needed |
| Data Validation | ✅ Full | ✅ Full | Logic-only, CPU sufficient |
| Dataset Creation | ✅ Full | ✅ Full | HF datasets work on CPU |
| NeMo Export | ⚠️ Partial | ✅ Full | Can test format, not tokenizer |
| Configuration | ✅ Full | ✅ Full | YAML parsing, validation |
| Price API | ⚠️ Partial | ✅ Full | Needs API key, rate limits |
| Logging | ✅ Full | ✅ Full | Pure Python |
| Manifest | ✅ Full | ✅ Full | Pure Python |
| Model Conversion | ❌ None | ✅ Full | Requires GPU (Triton kernels) |
| Training | ❌ None | ✅ Full | Requires 8×H100 |
| Evaluation | ⚠️ Logic Only | ✅ Full | Can test code, not inference |
| Backtesting | ⚠️ Logic Only | ✅ Full | Can test code, not real predictions |

**Coverage:** ~80% of codebase can be validated on CPU before GPU deployment.
