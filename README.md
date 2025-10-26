# NeMo-Charlie: DeepSeek-V3 Financial Trading Pipeline

A production-grade supervised fine-tuning pipeline for training DeepSeek-V3 models on financial trading data using NVIDIA NeMo Framework.

## 🚀 Quick Start

### Prerequisites

- **GPU:** 8×H100 80GB with NVLink (or similar)
- **OS:** Linux with CUDA 12.1+
- **Python:** 3.10+
- **Disk:** 500 GB recommended

### Installation

```bash
# Clone repository
git clone git@github.com:co1onnese/NeMo-Charlie.git
cd NeMo-Charlie

# Install NeMo environment
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
source venv/bin/activate
```

### One-Time Model Setup

Convert DeepSeek-V3 from FP8 to NeMo format:

```bash
# 1. Clone FP8 model
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base checkpoints/source/deepseek-v3

# 2. Convert & import
bash scripts/convert/convert_deepseek_v3.sh \
  --source checkpoints/source/deepseek-v3 \
  --output checkpoints/bf16/deepseek-v3

python3 scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo
```

### Training Pipeline

```bash
# 1. Prepare data
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py
python3 src/data/export_nemo_dataset.py \
  --dataset_dir data/hf_datasets/sft_dataset \
  --output_dir data/nemo/sft_dataset

# 2. Train
python3 src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/main

# 3. Evaluate
python3 src/eval/evaluate_nemo.py \
  --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo \
  --dataset data/nemo/sft_dataset \
  --results results/eval_results.csv

# 4. Backtest
python3 src/backtest/trading_backtest.py \
  --eval_jsonl results/eval_results.csv \
  --config configs/backtest_config.yaml \
  --out backtests/baseline.csv
```

Or use the automated pipeline:

```bash
bash scripts/run_full_pipeline.sh
```

## 📁 Project Structure

```
NeMo-Charlie/
├── configs/
│   ├── nemo/
│   │   └── finetune.yaml          # NeMo training config
│   ├── backtest_config.yaml       # Backtest parameters
│   └── eval_config.json           # Evaluation metrics
├── data/
│   ├── raw_xml/                   # Input XML files
│   ├── jsonl/                     # Converted JSONL
│   ├── hf_datasets/               # Intermediate HF format
│   └── nemo/                      # NeMo-ready JSONL
├── src/
│   ├── parsers/
│   │   └── xml_to_jsonl.py        # XML→JSONL converter
│   ├── data/
│   │   ├── convert_dataset.py     # HF dataset creation
│   │   ├── export_nemo_dataset.py # NeMo JSONL export
│   │   └── price_data.py          # Financial data API
│   ├── train/
│   │   └── train_nemo.py          # NeMo training entrypoint
│   ├── eval/
│   │   ├── evaluate_nemo.py       # NeMo evaluation
│   │   └── metrics.py             # Metrics computation
│   ├── backtest/
│   │   └── trading_backtest.py    # Portfolio simulation
│   └── utils/
│       ├── logger.py              # Logging utilities
│       ├── manifest.py            # Reproducibility tracking
│       └── validation.py          # Data validation
├── scripts/
│   ├── convert/
│   │   ├── fp8_cast_bf16.py       # FP8→BF16 conversion
│   │   ├── kernel.py              # Triton kernels
│   │   ├── convert_deepseek_v3.sh # Wrapper script
│   │   └── import_to_nemo.py      # NeMo import
│   ├── setup_env.sh               # Environment setup
│   └── run_full_pipeline.sh       # Full pipeline
└── checkpoints/
    ├── source/                    # Downloaded FP8 model
    ├── bf16/                      # Converted BF16
    └── nemo/                      # NeMo .nemo archives
```

## 🎯 Features

- **NeMo Framework Integration:** Native support for DeepSeek-V3 with Megatron-Core parallelism
- **Full-Parameter Fine-Tuning:** Train entire model on 8×H100, not just adapters
- **Long Context:** Supports up to 131k tokens with efficient attention
- **Time-Based Splits:** Prevents data leakage with strict chronological separation
- **Comprehensive Evaluation:** Both NLP accuracy and financial performance metrics
- **Portfolio Backtesting:** Realistic simulation with transaction costs and slippage
- **Reproducibility:** Complete manifests with git commits, configs, and data checksums

## 📊 Pipeline Stages

### 1. Data Processing

- Parse XML thesis files to JSONL
- Create time-based train/validation/test splits
- Export to NeMo-compatible JSONL format
- Add special tokens for structured output

### 2. Model Conversion

- Download DeepSeek-V3 FP8 checkpoint
- Convert to BF16 using Triton kernels
- Import into NeMo `.nemo` archive format
- Configure for 8-way tensor parallelism

### 3. Training

- Full-parameter fine-tuning or LoRA
- Distributed across 8×H100 with NVLink
- Support for sequences up to 65k tokens
- WandB/TensorBoard monitoring

### 4. Evaluation & Backtesting

- Generate predictions on test set
- Compute classification metrics (accuracy, F1)
- Calculate financial metrics (hit rate, Sharpe)
- Simulate portfolio performance

## ⚙️ Configuration

### Environment Variables (`.env`)

```bash
# API Keys
EODHD_API_KEY=your_key

# Date Ranges
TRAIN_END_DATE=2024-12-31
TEST_START_DATE=2025-01-01

# Training Backend
TRAIN_BACKEND=nemo
```

### NeMo Training Config (`configs/nemo/finetune.yaml`)

```yaml
recipe:
  factory: deepseek_v3
  resume_path: checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo

train:
  peft: none               # Full fine-tune
  seq_length: 65536
  micro_batch_size: 1
  global_batch_size: 128
  num_nodes: 1
  gpus_per_node: 8
```

## 📈 Expected Performance

- **Action Accuracy:** 60-75%
- **Hit Rate:** 55-60%
- **Sharpe Ratio:** 0.5-1.0

## 🔧 Hardware Requirements

### Development
- 1×A100/H100 for testing
- 100 GB disk

### Production
- 8×H100 80GB (single node, NVLink)
- 512 GB RAM
- 500 GB NVMe storage

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Fast onboarding guide
- **[NEMO_MIGRATION.md](NEMO_MIGRATION.md)** - Migration from TRL to NeMo
- **[runbook/README.md](runbook/README.md)** - Complete technical documentation
- **[MODEL_STORAGE_GUIDE.md](MODEL_STORAGE_GUIDE.md)** - Storage requirements and layout

## 🛠️ Troubleshooting

### NeMo Import Fails

Ensure CUDA 12.1+ and compatible PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Training OOM

Reduce sequence length or enable LoRA:
```yaml
train:
  peft: lora
  seq_length: 32768
```

### Evaluation Errors

Check that NeMo JSONL files have correct structure:
```bash
head -1 data/nemo/sft_dataset/training.jsonl | python3 -m json.tool
```

## 🤝 Contributing

This pipeline is designed for financial research use. When extending:

1. Maintain time-based split validation
2. Add tests for new components
3. Update manifest generation for reproducibility
4. Document hardware requirements

## 📄 License

See LICENSE files for model and code licensing.

## 🔗 Links

- **NeMo Framework:** https://github.com/NVIDIA/NeMo
- **DeepSeek-V3:** https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
- **Documentation:** https://docs.nvidia.com/nemo-framework/

## Migration Notes

This project was migrated from TRL/Hugging Face to NVIDIA NeMo in October 2025. Legacy scripts are retained but deprecated:

- ❌ `src/train/train_sft.py` → Use `src/train/train_nemo.py`
- ❌ `src/eval/evaluate_sft.py` → Use `src/eval/evaluate_nemo.py`
- ❌ `src/data/tokenize_and_shard.py` → Use `src/data/export_nemo_dataset.py`

See [NEMO_MIGRATION.md](NEMO_MIGRATION.md) for full migration details.
