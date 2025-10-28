# NeMo-Charlie: DeepSeek-V3 Financial Trading Pipeline

A production-grade supervised fine-tuning pipeline for training DeepSeek-V3 models on financial trading data using NVIDIA NeMo Framework. This pipeline processes financial thesis data, trains models with full-parameter fine-tuning or LoRA, and evaluates performance using both NLP metrics and portfolio backtesting.

## Table of Contents

- [Quick Start (Fresh Server)](#quick-start-fresh-server)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Model Conversion](#model-conversion)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation & Backtesting](#evaluation--backtesting)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Quick Start (Fresh Server)

**One-command setup that just works:**

```bash
# 1. Clone and navigate
cd /opt
git clone https://github.com/yourorg/NeMo-Charlie.git
cd NeMo-Charlie

# 2. Configure
cp .env.example .env
nano .env  # Set CPU_ONLY_MODE=true (dev) or false (GPU server)

# 3. Run setup (fully automated, 10-15 minutes)
bash scripts/setup_env_v2.sh

# 4. Activate and start working
source venv/bin/activate
python scripts/convert/import_to_nemo.py --help
```

**That's it!** Setup includes:
- ✅ Virtual environment creation
- ✅ PyTorch installation (CPU or GPU+CUDA 12.8)
- ✅ Optimal dependency installation order
- ✅ NeMo Framework with automatic patching
- ✅ Comprehensive validation
- ✅ **Minimal warnings** (only expected "NeMo-Run" warning)

### Expected Results

**CPU Mode** (`CPU_ONLY_MODE=true`):
```
✓ PyTorch 2.9.0 (CPU)
✓ Zarr 2.x (checkpoint format support)
✓ NeMo 2.5.2 (patched)
✓ All dependencies installed

⚠️ Expected warning: "NeMo-Run is not installed" (harmless, confirms patches work)
✅ Time: ~5-10 minutes
```

**GPU Mode** (`CPU_ONLY_MODE=false`):
```
✓ PyTorch 2.9.0+cu128 (GPU)
✓ Zarr 2.x
✓ Transformer Engine (20-40% faster training!)
✓ NeMo 2.5.2 (patched)
✓ All dependencies installed

⚠️ Expected warning: "NeMo-Run is not installed" (harmless, confirms patches work)
✅ Time: ~10-15 minutes
```

### What Gets Installed

The setup script installs dependencies in **optimal order** to minimize warnings:

1. **PyTorch** (foundation)
2. **zarr 2.x** - installed **BEFORE** NeMo (eliminates "Cannot import zarr" warning)
3. **transformer-engine** - installed **BEFORE** NeMo, GPU only (eliminates "transformer_engine" warning)
4. **Base requirements** (transformers, datasets, pandas, etc.)
5. **NeMo Framework** (detects already-installed optional deps)
6. **Automatic patches** (handles nemo_run gracefully)
7. **Validation** (comprehensive environment check)

## Prerequisites

### Hardware

**Minimum (Development/Testing):**
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 100 GB
- GPU: Optional (for faster operations)

**Production (Full Training):**
- GPUs: **8×H100 80GB** with NVLink (single node)
- RAM: 512 GB
- Disk: 500 GB NVMe storage
- CUDA: 12.4+

### Software

- **OS:** Ubuntu 20.04+ (tested on 22.04)
- **Python:** 3.12 (3.10+ supported)
- **Git:** For cloning repository
- **CUDA Toolkit:** 12.x (for GPU mode)

### Before You Start

```bash
# Ubuntu/Debian - Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    build-essential

# For GPU mode - verify CUDA is installed
nvidia-smi  # Should show CUDA 12.x
```

## Installation

### Step 1: Clone Repository

```bash
cd /opt
git clone https://github.com/yourorg/NeMo-Charlie.git
cd NeMo-Charlie
```

### Step 2: Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Critical setting in `.env`:**
```bash
# For CPU-only development/testing
CPU_ONLY_MODE=true

# For GPU training (requires CUDA 12.x)
CPU_ONLY_MODE=false
```

**Other important settings:**
```bash
# API key for price data (optional)
EODHD_API_KEY=your_key_here

# Date ranges for train/test splits
TRAIN_END_DATE=2024-12-31
TEST_START_DATE=2025-01-01
```

### Step 3: Run Setup

```bash
bash scripts/setup_env_v2.sh
```

**What happens:**
1. Checks Python 3.12+ and prerequisites
2. Reads `CPU_ONLY_MODE` from `.env`
3. Verifies CUDA (if GPU mode)
4. **Fails if `venv/` exists** (ensures clean install)
5. Creates virtual environment
6. Installs PyTorch (CPU or GPU+cu128)
7. **Installs zarr 2.x BEFORE NeMo** (key optimization!)
8. Installs transformer-engine (GPU mode only)
9. Installs all base requirements
10. Installs NeMo Framework
11. Applies patches automatically
12. **Runs comprehensive validation**

**Time:** 5-10 minutes (CPU), 10-15 minutes (GPU)

**Fully automated** - no prompts, no interaction needed!

### Step 4: Activate Environment

```bash
source venv/bin/activate
```

**You're ready to go!** The validation already ran during setup.

### Troubleshooting Setup

**Problem: "venv/ already exists"**
```bash
rm -rf venv/
bash scripts/setup_env_v2.sh
```

**Problem: ".env file not found"**
```bash
cp .env.example .env
# Edit and set CPU_ONLY_MODE
bash scripts/setup_env_v2.sh
```

**Problem: "CUDA not found" (GPU mode)**
```bash
# Either install CUDA 12.x, or switch to CPU mode
nano .env  # Set CPU_ONLY_MODE=true
bash scripts/setup_env_v2.sh
```

**Problem: "Python 3.12 not found"**
```bash
# Ubuntu 22.04+
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

### Migrating Between Environments

**From Dev (CPU) to Production (GPU):**
```bash
# On GPU server
cd /opt/NeMo-Charlie

# Update .env
sed -i 's/CPU_ONLY_MODE=true/CPU_ONLY_MODE=false/' .env

# Clean reinstall
rm -rf venv/
bash scripts/setup_env_v2.sh
```

### Manual Validation

If you want to re-run validation later:
```bash
source venv/bin/activate
python scripts/validate_environment.py
```

## Model Conversion

DeepSeek-V3 requires one-time conversion from FP8 to NeMo format.

### Step 1: Download Model

```bash
# Clone FP8 model from HuggingFace (requires git-lfs)
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base checkpoints/source/deepseek-v3
```

**Size:** ~1.3 TB (671B parameters in FP8)

### Step 2: Convert FP8 → BF16

```bash
bash scripts/convert/convert_deepseek_v3.sh \
  --source checkpoints/source/deepseek-v3 \
  --output checkpoints/bf16/deepseek-v3
```

The script automatically selects the best conversion mode based on your hardware:
- **Multi-GPU mode** (8 GPUs): ~4-6 minutes
- **Single-GPU mode** (1 GPU): ~10-15 minutes
- Uses Triton kernels for efficient FP8 dequantization
- Optimized with async I/O and parallel processing

**Requirements:**
- ~40GB GPU memory per GPU
- For multi-GPU: Requires 4+ GPUs and `torchrun`

**Advanced options:**
```bash
# Force specific mode
CONVERSION_MODE=multi ./convert_deepseek_v3.sh --source ... --output ...
# Modes: auto (default), single, multi

# Adjust I/O workers for performance tuning
./convert_deepseek_v3.sh --source ... --output ... --max-workers 16
```

### Step 3: Import to NeMo

```bash
python scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo \
  --tensor-parallel 8 \
  --pipeline-parallel 1
```

This creates a `.nemo` archive optimized for 8×H100.

**Requirements:**
- Must run with: `torchrun --nproc_per_node=8`
- ~10-30 minutes

## Data Preparation

### Pipeline Architecture

```
Raw XML Files
    ↓
XML Parser → JSONL Records
    ↓
Dataset Converter → Time-based Train/Val/Test Splits
    ↓
HuggingFace Dataset (intermediate)
    ↓
NeMo Export → training/validation/test.jsonl
```

### Step 1: Parse XML Data

```bash
# Place XML files in data/raw_xml/
cp /path/to/*.xml data/raw_xml/

# Parse to JSONL
python src/parsers/xml_to_jsonl.py \
  --input_dir data/raw_xml \
  --output_file data/jsonl/all.jsonl \
  --validate
```

**Output:** Normalized JSONL with fields: `instruction`, `input`, `output`, `date`, `ticker`, etc.

### Step 2: Create Dataset Splits

```bash
python src/data/convert_dataset.py \
  --jsonl data/jsonl/all.jsonl \
  --out_dir data/hf_datasets/sft_dataset \
  --train_end 2024-12-31 \
  --test_start 2025-01-01 \
  --validation_days 30
```

**Features:**
- Time-based splits (prevents data leakage)
- Validation period from train split
- Comprehensive statistics and metadata

### Step 3: Export to NeMo Format

```bash
python src/data/export_nemo_dataset.py \
  --dataset_dir data/hf_datasets/sft_dataset \
  --output_dir data/nemo/sft_dataset \
  --template chatml \
  --include_metadata
```

**Templates available:**
- `chatml`: ChatML format with special tokens
- `alpaca`: Alpaca instruction format
- `simple`: Plain concatenation

**Output:** Three files in `data/nemo/sft_dataset/`:
- `training.jsonl`
- `validation.jsonl`
- `test.jsonl`

## Training

### Configuration

Edit `configs/nemo/finetune.yaml`:

```yaml
recipe:
  factory: deepseek_v3
  name: deepseek_v3_finetune
  resume_path: checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo

dataset:
  path: data/nemo/sft_dataset
  template: chatml
  label_key: output
  answer_only_loss: true

train:
  peft: none                 # 'none' for full, 'lora' for adapters
  seq_length: 65536          # Max sequence length
  micro_batch_size: 1        # Per-GPU batch size
  global_batch_size: 128     # Total batch size
  num_nodes: 1
  gpus_per_node: 8
```

### Smoke Test (Recommended)

Test training with 10 steps:

```bash
python src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/smoke \
  --smoke-test
```

**Time:** ~5-10 minutes

### Full Training

```bash
python src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/main
```

**Outputs:**
- Fine-tuned checkpoint: `checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo`
- Training manifest: `manifest.json` (git hash, config, data checksums)
- Logs: `logs/train_*.log`

### Training Options

**Full-Parameter Fine-Tuning:**
```yaml
train:
  peft: none
```
- Requires 8×H100 80GB
- Best accuracy
- Slowest training

**LoRA (Low-Rank Adaptation):**
```yaml
train:
  peft: lora
```
- Requires fewer GPUs
- Faster training
- Slightly lower accuracy

## Evaluation & Backtesting

### Step 1: Generate Predictions

```bash
python src/eval/evaluate_nemo.py \
  --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo \
  --dataset data/nemo/sft_dataset \
  --split test \
  --results results/eval_results.csv \
  --metrics-json results/metrics.json
```

**Outputs:**
- `eval_results.csv`: Predictions with true/predicted actions
- `metrics.json`: Classification and financial metrics

**Metrics computed:**
- **NLP Metrics:** Accuracy, Precision, Recall, F1
- **Financial Metrics:** Hit Rate, Sharpe Ratio, Returns

### Step 2: Portfolio Backtesting

```bash
python src/backtest/trading_backtest.py \
  --eval_jsonl results/eval_results.csv \
  --config configs/backtest_config.yaml \
  --out backtests/baseline.csv
```

**Configuration** (`configs/backtest_config.yaml`):
```yaml
capital: 100000
transaction_cost: 0.001
slippage: 0.0005
sizing_strategy: equal_weight
```

**Outputs:**
- `backtests/baseline.csv`: Equity curve over time
- `backtests/baseline_metrics.json`: Performance statistics

### Expected Performance

- **Action Accuracy:** 60-75%
- **Hit Rate:** 55-60%
- **Sharpe Ratio:** 0.5-1.0

## Configuration

### NeMo Training Parameters

**Key settings in `configs/nemo/finetune.yaml`:**

```yaml
# Model parallelism
recipe:
  tensor_model_parallel_size: 8
  pipeline_model_parallel_size: 1

# Sequence handling
train:
  seq_length: 65536           # Max tokens per sample
  micro_batch_size: 1         # Samples per GPU
  global_batch_size: 128      # Total batch size

# Optimization
  max_steps: 1000
  val_check_interval: 100
  log_every_n_steps: 10

# LoRA settings (if peft: lora)
  lora_rank: 32
  lora_alpha: 64
  lora_dropout: 0.05
```

### Environment Variables

Key settings in `.env`:

```bash
# Hardware mode (CRITICAL)
CPU_ONLY_MODE=false          # true for dev, false for GPU training

# API keys
EODHD_API_KEY=your_key_here  # For price data (optional)

# Date configuration
TRAIN_END_DATE=2024-12-31
VALIDATION_DAYS=30
TEST_START_DATE=2025-01-01

# Paths (optional, defaults shown)
RAW_XML_DIR=data/raw_xml
JSONL_OUTPUT=data/jsonl/all.jsonl
HF_DATASET_DIR=data/hf_datasets/sft_dataset
NEMO_DATASET_DIR=data/nemo/sft_dataset

# Monitoring (optional)
USE_WANDB=false
WANDB_PROJECT=DeepSeek_Trading
```

## Project Structure

```
NeMo-Charlie/
├── configs/
│   ├── nemo/
│   │   └── finetune.yaml          # NeMo training config
│   ├── backtest_config.yaml       # Backtest parameters
│   └── eval_config.json           # Evaluation metrics
│
├── data/
│   ├── raw_xml/                   # Input XML files
│   ├── jsonl/                     # Parsed JSONL
│   ├── hf_datasets/               # HuggingFace format
│   └── nemo/                      # NeMo-ready JSONL
│
├── src/
│   ├── parsers/
│   │   └── xml_to_jsonl.py        # XML→JSONL converter
│   ├── data/
│   │   ├── convert_dataset.py     # Dataset creation
│   │   ├── export_nemo_dataset.py # NeMo export
│   │   └── price_data.py          # Price data API
│   ├── train/
│   │   └── train_nemo.py          # Training entrypoint
│   ├── eval/
│   │   ├── evaluate_nemo.py       # Evaluation
│   │   └── metrics.py             # Metrics computation
│   ├── backtest/
│   │   └── trading_backtest.py    # Portfolio simulation
│   └── utils/
│       ├── logger.py              # Logging
│       ├── manifest.py            # Reproducibility
│       └── validation.py          # Data validation
│
├── scripts/
│   ├── setup_env_v2.sh            # ONE-COMMAND SETUP ⭐
│   ├── validate_environment.py    # Validation (auto-run by setup)
│   ├── run_full_pipeline.sh       # Full pipeline automation
│   ├── apply_nemo_patches.py      # NeMo patching (auto-run)
│   └── convert/
│       ├── convert_deepseek_v3.sh # FP8→BF16 wrapper
│       ├── fp8_cast_bf16.py       # FP8→BF16 conversion
│       ├── kernel.py              # Triton kernels
│       └── import_to_nemo.py      # NeMo import
│
└── checkpoints/
    ├── source/                    # Downloaded FP8 model
    ├── bf16/                      # Converted BF16
    ├── nemo/                      # Base .nemo archives
    └── nemo_runs/                 # Fine-tuned checkpoints
```

## Troubleshooting

### Environment Issues

**Problem: Warnings about missing packages**

Check which warnings you see:

✅ **"NeMo-Run is not installed"** → Expected and harmless! This confirms patches are working. nemo_run doesn't exist on public PyPI.

❌ **"Cannot import zarr"** → Should NOT appear with setup_env_v2.sh. If you see this:
```bash
source venv/bin/activate
pip show zarr  # Check version (should be 2.x not 3.x)
pip uninstall zarr && pip install "zarr>=2.16.0,<3.0.0"
```

❌ **"transformer_engine not installed"** in GPU mode → Should NOT appear. Check:
```bash
pip show transformer-engine
# If missing:
pip install transformer-engine[pytorch]
```

✅ **"transformer_engine not installed"** in CPU mode → Expected and harmless!

**Problem: "No module named 'nv_one_logger'"**

Patches weren't applied. Should not happen with setup_env_v2.sh, but if it does:
```bash
source venv/bin/activate
python scripts/apply_nemo_patches.py
bash scripts/verify_nemo_fixes.sh
```

**Problem: Need to start fresh**
```bash
rm -rf venv/
bash scripts/setup_env_v2.sh
```

### Conversion Issues

**Problem: FP8→BF16 conversion OOM**

Insufficient GPU memory (~40GB required).

**Solution:**
- Use H100 or A100 80GB
- Close other GPU processes
- Check: `nvidia-smi`

**Problem: Import to NeMo fails**

Must run with torchrun for multi-GPU.

**Solution:**
```bash
torchrun --nproc_per_node=8 scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo
```

### Training Issues

**Problem: Training OOM**

Model too large for available GPU memory.

**Solutions:**
1. Reduce sequence length:
   ```yaml
   train:
     seq_length: 32768  # Down from 65536
   ```

2. Use LoRA instead of full fine-tuning:
   ```yaml
   train:
     peft: lora
   ```

3. Reduce batch size:
   ```yaml
   train:
     micro_batch_size: 1
     global_batch_size: 64  # Down from 128
   ```

**Problem: "CUDA out of memory" during training**

**Solution:**
```bash
# Clear GPU memory
nvidia-smi --gpu-reset

# Check GPU utilization
nvidia-smi dmon
```

**Problem: Training is very slow**

**Solutions:**
1. Enable performance mode:
   ```yaml
   train:
     performance_mode: true
   ```

2. Use packed sequences:
   ```yaml
   dataset:
     packed_sequence_size: 65536
   ```

### Evaluation Issues

**Problem: "Dataset file not found"**

Check NeMo JSONL structure.

**Solution:**
```bash
# Verify files exist
ls -lh data/nemo/sft_dataset/

# Check JSON structure
head -1 data/nemo/sft_dataset/test.jsonl | python -m json.tool
```

**Problem: Price data API failures**

Rate limiting or invalid API key.

**Solution:**
```bash
# Check API key in .env
grep EODHD_API_KEY .env

# Test API manually
curl "https://eodhd.com/api/eod/AAPL.US?api_token=YOUR_KEY"
```

## Advanced Topics

### Monitoring with WandB

```bash
# Set in .env
USE_WANDB=true
WANDB_PROJECT=DeepSeek_Trading

# Login
wandb login
```

Metrics will be logged to W&B dashboard during training.

### Multi-Node Training

For very large models or datasets:

```yaml
train:
  num_nodes: 2
  gpus_per_node: 8
```

Launch with SLURM or manual coordination across nodes.

### Custom Data Formats

To use your own data:

1. Convert to JSONL with fields: `instruction`, `input`, `output`
2. Run `convert_dataset.py` with your JSONL
3. Export with `export_nemo_dataset.py`
4. Train normally

### Reproducibility

Every training run creates a manifest with:
- Git commit hash
- Full config
- Data file checksums
- Environment details

Located at: `checkpoints/nemo_runs/<run_name>/manifest.json`

### Full Pipeline Automation

```bash
# Run entire pipeline (data → train → eval → backtest)
bash scripts/run_full_pipeline.sh

# With smoke test mode
bash scripts/run_full_pipeline.sh --smoke-test

# Skip certain stages
bash scripts/run_full_pipeline.sh --skip-train --skip-eval
```

## Technical Details

### Why Installation Order Matters

NeMo checks for optional packages **at import time**. If you install them after NeMo, warnings appear even though they're present. Our setup script installs:

1. PyTorch (foundation)
2. **zarr 2.x** - BEFORE NeMo (eliminates warning)
3. **transformer-engine** - BEFORE NeMo, GPU only (eliminates warning)
4. NeMo (detects already-installed packages)

### Why zarr < 3.0.0

NeMo 2.5.2 uses `zarr.storage.BaseStore` which only exists in zarr 2.x. Version 3.x has breaking API changes.

### About the NeMo-Run Warning

The "NeMo-Run is not installed" warning **confirms your patches are working correctly**. It's expected and harmless because:
- `nemo_run` doesn't exist on public PyPI (internal NVIDIA tool)
- Patches make it optional with graceful fallback
- All core functionality works without it

If you don't see this warning, patches may not be applied.

### About the Patches

NeMo 2.5.2 has hardcoded imports for optional dependencies:
- `nv_one_logger` (telemetry - not on PyPI)
- `nemo_run` (recipes - not on PyPI)
- `tensorstore` (export - optional)

Our patches (`apply_nemo_patches.py`) make these imports conditional with graceful fallbacks. See `NEMO_FIXES.md` for detailed patch documentation.

## Support & Documentation

- **NeMo Framework:** https://github.com/NVIDIA/NeMo
- **DeepSeek-V3:** https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
- **NeMo Documentation:** https://docs.nvidia.com/nemo-framework/
- **Patch Details:** See `NEMO_FIXES.md` in this repository

## Features

- ✅ **One-Command Setup** - Fully automated installation with validation
- ✅ **Minimal Warnings** - Optimal installation order eliminates unnecessary warnings
- ✅ **Configuration-Driven** - CPU vs GPU mode via `.env`
- ✅ **Native NeMo Integration** - Full DeepSeek-V3 support with Megatron-Core
- ✅ **Full-Parameter Fine-Tuning** - Train entire 671B model, not just adapters
- ✅ **Long Context** - Supports up to 131k tokens with efficient attention
- ✅ **Time-Based Splits** - Prevents data leakage with chronological separation
- ✅ **Comprehensive Evaluation** - Both NLP and financial metrics
- ✅ **Portfolio Backtesting** - Realistic simulation with costs and slippage
- ✅ **Reproducibility** - Complete manifests with git, configs, and checksums
- ✅ **Comprehensive Validation** - Automatic environment verification

## License

See LICENSE files for model and code licensing.
