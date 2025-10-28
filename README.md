# NeMo-Charlie: DeepSeek-V3 Financial Trading Pipeline

A production-grade supervised fine-tuning pipeline for training DeepSeek-V3 models on financial trading data using NVIDIA NeMo Framework. This pipeline processes financial thesis data, trains models with full-parameter fine-tuning or LoRA, and evaluates performance using both NLP metrics and portfolio backtesting.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Model Conversion](#model-conversion)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation & Backtesting](#evaluation--backtesting)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Prerequisites

### Hardware

**Minimum (Development/Testing):**
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 100 GB
- GPU: Optional (1×A100/H100 for testing)

**Production (Full Training):**
- GPUs: **8×H100 80GB** with NVLink (single node)
- RAM: 512 GB
- Disk: 500 GB NVMe storage
- CUDA: 12.4+

### Software

- **OS:** Linux (Ubuntu 20.04+ recommended)
- **Python:** 3.8+ (3.10 or 3.12 recommended)
- **Git:** For cloning repository
- **CUDA Toolkit:** 12.4+ (for GPU training)

## Quick Start

### 1. Installation

Clone the repository and run the automated setup script:

```bash
# Clone repository
git clone https://github.com/co1onnese/NeMo-Charlie.git
cd NeMo-Charlie

# Run comprehensive setup (installs everything + applies patches)
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

**This single command:**
1. ✅ Creates Python virtual environment
2. ✅ Installs PyTorch with CUDA 12.4 support
3. ✅ Installs all dependencies
4. ✅ Installs NeMo Framework
5. ✅ Automatically applies NeMo patches
6. ✅ Verifies everything works

**Time:** ~10-15 minutes

### 2. Activate Environment

Every time you start a new terminal session:

```bash
cd NeMo-Charlie
source venv/bin/activate
```

### 3. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

Set these variables:
```bash
# API Keys
EODHD_API_KEY=your_api_key

# Date Ranges
TRAIN_END_DATE=2024-12-31
TEST_START_DATE=2025-01-01
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

This uses Triton kernels for efficient tensor conversion.

**Requirements:**
- ~40GB GPU memory
- ~20-30 minutes on H100

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

Create `.env` file:

```bash
# Required
EODHD_API_KEY=your_key_here

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
│   ├── setup_env.sh               # Environment setup
│   ├── run_full_pipeline.sh       # Full pipeline automation
│   ├── apply_nemo_patches.py      # NeMo patching (auto)
│   ├── verify_nemo_fixes.sh       # Verify patches
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

### Setup Issues

**Problem: "No module named 'nv_one_logger'"**

NeMo patches weren't applied.

**Solution:**
```bash
source venv/bin/activate
python scripts/apply_nemo_patches.py
```

**Problem: Setup script hangs**

Large downloads (~2GB PyTorch, ~1GB NeMo).

**Solution:**
- Be patient, monitor with `htop`
- Check disk space: `df -h`
- Check network: `ping pypi.org`

**Problem: "pip install failed"**

Network issues or missing dependencies.

**Solution:**
```bash
# Try with verbose output
pip install -r requirements_nemo.txt -v

# Install core packages individually
pip install nemo-toolkit megatron-core lightning
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

### General

**Problem: Need to start fresh**

Environment corrupted or wants clean install.

**Solution:**
```bash
# Remove virtual environment
rm -rf venv/

# Re-run setup
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

**Problem: Verify patches are working**

After updating NeMo or dependencies.

**Solution:**
```bash
source venv/bin/activate
bash scripts/verify_nemo_fixes.sh
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

## Support & Documentation

- **NeMo Framework:** https://github.com/NVIDIA/NeMo
- **DeepSeek-V3:** https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
- **NeMo Documentation:** https://docs.nvidia.com/nemo-framework/
- **Technical Details:** See `NEMO_FIXES.md` for patch documentation

## Features

- ✅ **Native NeMo Integration** - Full DeepSeek-V3 support with Megatron-Core
- ✅ **Full-Parameter Fine-Tuning** - Train entire 671B model, not just adapters
- ✅ **Long Context** - Supports up to 131k tokens with efficient attention
- ✅ **Time-Based Splits** - Prevents data leakage with chronological separation
- ✅ **Comprehensive Evaluation** - Both NLP and financial metrics
- ✅ **Portfolio Backtesting** - Realistic simulation with costs and slippage
- ✅ **Reproducibility** - Complete manifests with git, configs, and checksums
- ✅ **Automated Setup** - Single command installation with auto-patching

## License

See LICENSE files for model and code licensing.
