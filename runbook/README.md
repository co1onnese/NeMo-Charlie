# SFT Trading Pipeline - Phase 1 Documentation

## Overview

This is a complete supervised fine-tuning (SFT) pipeline for training financial trading models using DeepSeek-V3.2-Exp. The pipeline processes XML thesis data, trains a model to generate structured reasoning and trading actions, and evaluates performance using both NLP metrics and financial backtesting.

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
Tokenizer (tokenize_and_shard.py) → Special Tokens for XML/Actions
    ↓
HuggingFace Dataset
    ↓
SFT Trainer (train_sft.py) → PEFT/LoRA + QLoRA
    ↓
Fine-tuned Model
    ↓
Evaluator (evaluate_sft.py) → NLP + Financial Metrics
    ↓
Backtester (trading_backtest.py) → Portfolio Simulation
    ↓
Results & Analysis
```

## Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to repository
cd /opt/SFT-Charlie

# Run setup script
bash scripts/setup_env.sh

# Activate virtual environment
source venv/bin/activate

# Verify installation
python3 -c "import torch, transformers, datasets, trl, peft; print('✓ All imports successful')"
```

### 2. Configuration

The pipeline uses `.env` for configuration. Key settings:

```bash
# API Keys
EODHD_API_KEY=your_api_key_here

# Date Ranges
TRAIN_START_DATE=2023-10-24
TRAIN_END_DATE=2024-12-31
TEST_START_DATE=2025-01-01
TEST_END_DATE=2025-04-24

# Model
BASE_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
MAX_LENGTH=65536
```

Edit `.env` to customize settings.

### 3. Prepare Data

```bash
# Copy your XML files to data/raw_xml/
cp /path/to/your/*.xml data/raw_xml/

# Run data pipeline
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py
python3 src/data/tokenize_and_shard.py
```

### 4. Train Model (GPU Required)

```bash
# Single GPU
python3 src/train/train_sft.py --config configs/sft_config.yaml

# Smoke test (10 steps)
python3 src/train/train_sft.py --config configs/sft_config.yaml --smoke_test

# Multi-GPU with DeepSpeed
deepspeed --num_gpus=4 src/train/train_sft.py --config configs/sft_config_multigpu.yaml
```

### 5. Evaluate Model

```bash
# Generate predictions and compute metrics
python3 src/eval/evaluate_sft.py \
    --model_dir checkpoints/sft-deepseek-v3.2exp-longctx \
    --dataset_dir data/hf_datasets/sft_dataset \
    --out results/eval_results.csv

# Run backtest simulation
python3 src/backtest/trading_backtest.py \
    --eval_csv results/eval_results.csv \
    --config configs/backtest_config.yaml \
    --out backtests/baseline.csv
```

## Directory Structure

```
/opt/SFT-Charlie/
├── data/
│   ├── raw_xml/              # Your XML thesis files
│   ├── jsonl/                # Converted JSONL
│   ├── hf_datasets/          # HuggingFace datasets
│   ├── price_cache/          # Cached price data
│   └── samples/              # Example/test data
├── src/
│   ├── parsers/              # XML parsing
│   ├── data/                 # Data processing
│   ├── train/                # Training scripts
│   ├── eval/                 # Evaluation
│   ├── backtest/             # Backtesting
│   └── utils/                # Utilities
├── configs/                  # YAML configurations
├── scripts/                  # Helper scripts
├── checkpoints/              # Saved models
├── results/                  # Evaluation outputs
├── backtests/                # Backtest results
├── logs/                     # Run logs
└── tests/                    # Test scripts
```

## Pipeline Components

### 1. XML Parser (`src/parsers/xml_to_jsonl.py`)

Converts XML thesis files to JSONL format.

**Input:** XML files with `<thesis>` tags containing `<reasoning>`, `<support>`, `<action>`

**Output:** JSONL file with one JSON object per thesis

**Features:**
- Validates XML structure
- Normalizes dates to ISO format
- Normalizes actions to standard values
- Handles missing/malformed data

### 2. Dataset Converter (`src/data/convert_dataset.py`)

Creates HuggingFace dataset with time-based splits.

**Features:**
- Strict chronological train/val/test splitting
- Prevents data leakage
- Validates data integrity
- Computes dataset statistics

**Time Splits:**
- Training: All data up to `TRAIN_END_DATE`
- Validation: Last N days of training period
- Test: From `TEST_START_DATE` onwards

### 3. Tokenizer (`src/data/tokenize_and_shard.py`)

Tokenizes dataset with special tokens for XML structure and actions.

**Special Tokens Added:**
- XML tags: `<reasoning>`, `</reasoning>`, `<support>`, `</support>`, `<action>`, `</action>`
- Actions: `<STRONG_BUY>`, `<BUY>`, `<HOLD>`, `<SELL>`, `<STRONG_SELL>`

### 4. SFT Trainer (`src/train/train_sft.py`)

Fine-tunes DeepSeek model using PEFT/LoRA and QLoRA.

**Features:**
- 4-bit quantization (QLoRA) for memory efficiency
- LoRA adapters for parameter-efficient training
- Gradient checkpointing for large context windows
- WandB integration (optional)
- Reproducibility manifests

**Key Hyperparameters:**
- Max length: 65536 tokens (long context)
- LoRA r: 8, alpha: 16
- Learning rate: 1.5e-4
- Batch size: 1 per device (with gradient accumulation)

### 5. Evaluator (`src/eval/evaluate_sft.py`)

Evaluates model on NLP and financial metrics.

**NLP Metrics:**
- Action classification accuracy
- Precision/Recall/F1 per class
- Confusion matrix

**Financial Metrics:**
- Hit rate (directional correctness)
- Mean/std realized returns
- Sharpe ratio estimation
- Per-action performance

**Price Data:**
- Uses eodhd.com API (primary)
- Falls back to yfinance
- Automatic caching

### 6. Backtester (`src/backtest/trading_backtest.py`)

Simulates portfolio performance based on model predictions.

**Features:**
- Configurable position sizing
- Transaction costs and slippage
- Daily rebalancing
- Risk metrics (Sharpe, Sortino, max drawdown)

## Testing

### Smoke Test (CPU Only)

Quick validation of data pipeline components:

```bash
bash scripts/smoke_test.sh
```

Tests:
- ✓ Validation utilities
- ✓ XML parsing
- ✓ Dataset conversion
- ✓ Price API (basic check)
- ⊘ Tokenization (skipped - requires model download)
- ⊘ Training (skipped - requires GPU)

### Full Pipeline Test

```bash
bash scripts/run_full_pipeline.sh --smoke-test
```

Runs entire pipeline with minimal data (10 training steps).

## Configuration Files

### `configs/sft_config.yaml`

Main training configuration for single GPU.

**Key Settings:**
```yaml
base_model: deepseek-ai/DeepSeek-V3.2-Exp
max_length: 65536
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.5e-4
lora_r: 8
```

### `configs/eval_config.yaml`

Evaluation configuration.

```yaml
forward_windows: [1, 5, 10, 30]
model_dir: checkpoints/sft-run
dataset_dir: data/hf_datasets/sft_dataset
```

### `configs/backtest_config.yaml`

Backtesting parameters.

```yaml
initial_cash: 1000000
transaction_cost_bps: 5
slippage_bps: 10
position_sizing: fixed_pct
fixed_pct_value: 0.02
```

## Data Requirements

### XML Format

Each XML file should contain thesis records:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<stock-theses ticker="TSLA" generated-by="Trainer-Charlie" version="1.0">
  <thesis>
    <as-of-date>2023-10-24</as-of-date>
    <reasoning>Detailed analysis...</reasoning>
    <action>buy</action>
    <support>Supporting points...</support>
  </thesis>
</stock-theses>
```

### Expected Data Volume

For your setup (~20 stocks, 548 days each):
- Total samples: ~10,960 thesis records
- Training set: ~10,050 (up to 2024-12-31)
- Validation set: ~365 (last 30 days of training)
- Test set: ~110 (2025-01-01 to 2025-04-24)

## Price Data

The pipeline fetches historical price data for evaluation:

**Primary Source:** eodhd.com API
- Requires API key (set in `.env`)
- Comprehensive coverage
- Rate-limited requests

**Fallback:** yfinance
- Free, no API key required
- May have gaps for some tickers
- Automatic fallback if eodhd fails

**Caching:**
- All price data cached in `data/price_cache/`
- Parquet format for fast loading
- Automatic cache updates

## Troubleshooting

### Common Issues

**1. "Dataset not found" error**
```bash
# Run data preparation steps
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py
```

**2. "CUDA out of memory"**
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_length`
- Enable `gradient_checkpointing`

**3. "Price data not available"**
- Check eodhd API key in `.env`
- Verify ticker symbols are correct
- Check date ranges are valid

**4. "Tokenizer errors"**
- Ensure `trust_remote_code=True` in config
- Try deleting HuggingFace cache: `~/.cache/huggingface`
- Re-download tokenizer

### Logs

All scripts write logs to `logs/` directory with timestamps.

Check latest log:
```bash
ls -lt logs/ | head -5
tail -f logs/[latest-log-file]
```

## GPU Server Deployment

When moving to GPU server:

1. **Clone repository:**
   ```bash
   git clone [repository-url] /path/on/gpu/server
   cd /path/on/gpu/server
   ```

2. **Copy .env file:**
   ```bash
   scp .env user@gpu-server:/path/on/gpu/server/
   ```

3. **Setup environment:**
   ```bash
   bash scripts/setup_env.sh
   # Install GPU version of PyTorch
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA version
   ```

4. **Copy data:**
   ```bash
   rsync -avz data/ user@gpu-server:/path/on/gpu/server/data/
   ```

5. **Update .env on GPU server:**
   ```bash
   # Set CPU_ONLY_MODE=false
   sed -i 's/CPU_ONLY_MODE=true/CPU_ONLY_MODE=false/' .env
   ```

6. **Run training:**
   ```bash
   python3 src/train/train_sft.py --config configs/sft_config.yaml
   ```

## Monitoring

### WandB Integration (Optional)

Enable in `.env`:
```bash
USE_WANDB=true
WANDB_PROJECT=DeepSeek_SFT_Trading
WANDB_ENTITY=your-entity
```

Login to WandB:
```bash
wandb login
```

### TensorBoard

Logs automatically saved to `{OUTPUT_DIR}/runs/`

View:
```bash
tensorboard --logdir checkpoints/sft-deepseek-v3.2exp-longctx/runs
```

## Reproducibility

Each training run creates a `manifest.json` with:
- Git commit hash
- Configuration hash
- Data file checksums
- Package versions
- Timestamp

This ensures full reproducibility of results.

## Next Steps (Phase 2)

After Phase 1 SFT is complete:

1. **GRPO RL Training** - Align model with trading rewards
2. **Trading Environment** - Custom RL environment with volatility-aware rewards
3. **Curriculum Learning** - Progressive difficulty in market scenarios
4. **Tool Integration** - External data sources and analysis tools

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review troubleshooting section
3. Verify `.env` configuration
4. Test with smoke test script

## License

[Add your license information here]
