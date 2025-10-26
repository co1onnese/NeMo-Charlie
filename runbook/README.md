# NeMo Trading Pipeline - Complete Documentation

## Overview

This is a complete supervised fine-tuning (SFT) pipeline for training financial trading models using DeepSeek-V3 with NVIDIA NeMo. The pipeline processes XML thesis data, trains a model via NeMo's full-parameter fine-tuning or LoRA, and evaluates performance using both NLP metrics and financial backtesting.

## Architecture

```
Raw XML Files
    ↓
XML Parser (xml_to_jsonl.py)
    ↓
JSONL Records
    ↓
Dataset Converter (convert_dataset.py) → Time-based Train/Val/Test Splits
    ↓
HuggingFace Dataset (intermediate format)
    ↓
NeMo Export (export_nemo_dataset.py) → training/validation/test.jsonl
    ↓
NeMo Fine-Tuning (train_nemo.py) → Full-parameter or LoRA
    ↓
Fine-tuned .nemo Checkpoint
    ↓
NeMo Evaluator (evaluate_nemo.py) → NLP + Financial Metrics
    ↓
Backtester (trading_backtest.py) → Portfolio Simulation
    ↓
Results & Analysis
```

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
cd /opt/SFT-Charlie

# Install NeMo environment
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate

# Verify NeMo installation
python3 -c "from nemo.collections import llm; print('✓ NeMo installed')"
```

### 2. Model Conversion (One-time Setup)

DeepSeek-V3 requires FP8→BF16 conversion before NeMo import:

```bash
# 1. Clone FP8 model from HuggingFace
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base checkpoints/source/deepseek-v3

# 2. Convert FP8→BF16
bash scripts/convert/convert_deepseek_v3.sh \
  --source checkpoints/source/deepseek-v3 \
  --output checkpoints/bf16/deepseek-v3

# 3. Import to NeMo archive
python3 scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo \
  --tensor-parallel 8 \
  --pipeline-parallel 1
```

### 3. Data Preparation

```bash
# Copy XML files
cp /path/to/your/*.xml data/raw_xml/

# Run data pipeline
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py \
  --jsonl data/jsonl/all.jsonl \
  --out_dir data/hf_datasets/sft_dataset

# Export NeMo-ready dataset
python3 src/data/export_nemo_dataset.py \
  --dataset_dir data/hf_datasets/sft_dataset \
  --output_dir data/nemo/sft_dataset \
  --template chatml \
  --include_metadata
```

### 4. Training (8×H100 Required)

```bash
# Smoke test (10 steps)
python3 src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/smoke \
  --smoke-test

# Full training
python3 src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/main
```

### 5. Evaluation & Backtesting

```bash
# Evaluate NeMo model
python3 src/eval/evaluate_nemo.py \
  --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo \
  --dataset data/nemo/sft_dataset \
  --split test \
  --results results/eval_results.csv

# Run backtest
python3 src/backtest/trading_backtest.py \
  --eval_jsonl results/eval_results.csv \
  --config configs/backtest_config.yaml \
  --out backtests/baseline.csv
```

## Configuration

### NeMo Training Config (`configs/nemo/finetune.yaml`)

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
  num_workers: 8

train:
  peft: none                 # 'none' for full fine-tune, 'lora' for adapters
  seq_length: 65536
  micro_batch_size: 1
  global_batch_size: 128
  num_nodes: 1
  gpus_per_node: 8
  performance_mode: false
```

### Environment Variables (`.env`)

```bash
# API Keys
EODHD_API_KEY=your_api_key

# Date Ranges
TRAIN_END_DATE=2024-12-31
TEST_START_DATE=2025-01-01

# Training Backend
TRAIN_BACKEND=nemo          # 'nemo' (default) or 'trl' (legacy)
```

## Pipeline Components

### 1. Model Conversion (`scripts/convert/`)

- `fp8_cast_bf16.py`: Converts FP8 weights to BF16 using Triton kernels
- `kernel.py`: Triton kernel utilities for dequantization
- `convert_deepseek_v3.sh`: Wrapper script for conversion
- `import_to_nemo.py`: Imports BF16 checkpoint into `.nemo` archive

### 2. Data Processing (`src/data/`)

- `convert_dataset.py`: Creates time-based HF dataset splits
- `export_nemo_dataset.py`: Exports HF dataset to NeMo JSONL format
- `price_data.py`: Fetches price data from eodhd.com API

### 3. Training (`src/train/`)

- `train_nemo.py`: NeMo-based fine-tuning entrypoint
- `train_sft.py`: Legacy TRL-based training (deprecated)

### 4. Evaluation (`src/eval/`)

- `evaluate_nemo.py`: NeMo checkpoint evaluation
- `metrics.py`: Classification and financial metrics computation
- `evaluate_sft.py`: Legacy HF evaluation (deprecated)

### 5. Utilities (`src/utils/`)

- `logger.py`: Centralized logging
- `manifest.py`: Reproducibility tracking
- `validation.py`: Data validation
- `eval_utils.py`: Evaluation helpers

## Hardware Requirements

### Minimum (Development/Testing)
- CPU: 8+ cores
- RAM: 32 GB
- Disk: 100 GB

### Production (Full Training)
- GPUs: 8×H100 80GB (single node with NVLink)
- RAM: 512 GB
- Disk: 500 GB
- Network: High-bandwidth interconnect for multi-node

## Troubleshooting

### NeMo Import Issues

```bash
# If NeMo import fails, check CUDA version
python3 -c "import torch; print(torch.version.cuda)"

# Ensure compatible PyTorch/CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Model Conversion Fails

```bash
# Check GPU memory during conversion
nvidia-smi

# Conversion requires ~40GB GPU memory
# If OOM, use a larger GPU or reduce batch processing
```

### Training OOM

Edit `configs/nemo/finetune.yaml`:
```yaml
train:
  micro_batch_size: 1
  seq_length: 32768  # Reduce from 65536
```

## Performance Optimization

### For LoRA Training

Set in config:
```yaml
train:
  peft: lora
  seq_length: 65536
```

This reduces GPU requirements from 8×H100 to potentially 5×H100.

### For Packed Sequences

```yaml
dataset:
  packed_sequence_size: 65536
  pad_cu_seqlens: false
```

## Monitoring

### WandB Integration

```bash
# Set in .env or config
USE_WANDB=true
WANDB_PROJECT=DeepSeek_SFT_Financial_Trading

# Login
wandb login
```

### Logs

All logs stored in `logs/` directory with timestamps.

```bash
# View latest log
tail -f logs/*.log

# Filter errors
grep ERROR logs/*.log
```

## Best Practices

1. **Always run smoke tests** before full training
2. **Monitor GPU memory** during conversion and training
3. **Backup checkpoints** regularly
4. **Validate data splits** to prevent leakage
5. **Use manifests** for reproducibility

## Migration Notes

This pipeline has been migrated from TRL/Hugging Face to NVIDIA NeMo for:
- Native DeepSeek-V3 support
- Better multi-GPU scaling
- Full-parameter fine-tuning capability
- Long-context optimization (131k tokens)

Legacy TRL scripts (`train_sft.py`, `evaluate_sft.py`) are retained but deprecated.
