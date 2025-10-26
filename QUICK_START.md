# Quick Start Guide - SFT Trading Pipeline

## 🚀 Immediate Next Steps

### On This CPU Server (Right Now)

1. **Copy Your XML Files**:
   ```bash
   cp /path/to/your/stock-thesis-files/*.xml /opt/SFT-Charlie/data/raw_xml/
   ```

2. **Convert XML → JSONL and export NeMo dataset** (works on CPU):
   ```bash
   cd /opt/SFT-Charlie
   python3 src/parsers/xml_to_jsonl.py
   python3 src/data/convert_dataset.py \
     --jsonl data/jsonl/all.jsonl \
     --out_dir data/hf_datasets/sft_dataset
   python3 src/data/export_nemo_dataset.py \
     --dataset_dir data/hf_datasets/sft_dataset \
     --output_dir data/nemo/sft_dataset \
     --tokenizer deepseek-ai/DeepSeek-V3.2-Exp \
     --tokenizer_out data/nemo/tokenizer
   ```
   
   Expected output: `data/jsonl/all.jsonl` with ~10,960 records

3. **Inspect Results**:
   ```bash
   # Count records
   wc -l data/jsonl/all.jsonl
   
   # View first record
   head -n 1 data/jsonl/all.jsonl | python3 -m json.tool
   ```

### When Ready for GPU Server

1. **Clone/Copy Project**:
   ```bash
   # On GPU server
   cd /desired/path
   git clone [your-repo] sft-charlie
   # OR
   rsync -avz /opt/SFT-Charlie/ gpu-server:/path/to/sft-charlie/
   ```

2. **Setup Environment**:
   ```bash
   cd sft-charlie
   bash scripts/setup_env.sh
   source venv/bin/activate
   
   # Install GPU PyTorch (replace cu118 with your CUDA version)
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Update Configuration**:
   ```bash
   # Edit .env file
   nano .env
   
   # Change this line:
   CPU_ONLY_MODE=false
   ```

4. **Run Full Pipeline (NeMo)**:
   ```bash
   bash scripts/run_full_pipeline.sh
   ```

   **Or step-by-step:**

   ```bash
   # 1. Export NeMo dataset (if not already done)
   python3 src/data/export_nemo_dataset.py \
     --dataset_dir data/hf_datasets/sft_dataset \
     --output_dir data/nemo/sft_dataset \
     --tokenizer deepseek-ai/DeepSeek-V3.2-Exp \
     --tokenizer_out data/nemo/tokenizer

   # 2. Smoke test training (10 steps)
   python3 src/train/train_nemo.py \
     --config configs/nemo/finetune.yaml \
     --output checkpoints/nemo_runs/smoke \
     --smoke-test

   # 3. Full training
   python3 src/train/train_nemo.py \
     --config configs/nemo/finetune.yaml \
     --output checkpoints/nemo_runs/main

   # 4. Evaluate (NeMo)
   python3 src/eval/evaluate_nemo.py \
     --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo \
     --dataset data/nemo/sft_dataset \
     --results results/eval_results.csv

   # 5. Backtest using NeMo results
   python3 src/backtest/trading_backtest.py \
     --eval_jsonl results/eval_results.csv \
     --config configs/backtest_config.yaml \
     --out backtests/baseline.csv
   ```

## 📁 File Locations

### Inputs:
- XML files: `data/raw_xml/*.xml`
- Configuration: `.env`, `configs/*.yaml`

### Outputs:
- JSONL data: `data/jsonl/all.jsonl`
- HF dataset: `data/hf_datasets/sft_dataset/`
- Trained model: `checkpoints/sft-deepseek-v3.2exp-longctx/`
- Evaluation: `results/eval_results.csv`
- Backtest: `backtests/baseline.csv`
- Logs: `logs/*.log`

## 🔧 Common Commands

### View Logs:
```bash
# Latest log
ls -lt logs/ | head -5
tail -f logs/[latest-log].log
```

### Check Data:
```bash
# Count samples in splits
python3 -c "
from datasets import load_from_disk
ds = load_from_disk('data/hf_datasets/sft_dataset')
for split in ds.keys():
    print(f'{split}: {len(ds[split])} samples')
"
```

### Monitor Training:
```bash
# If using WandB
wandb login
# Then check dashboard

# Or use TensorBoard
tensorboard --logdir checkpoints/sft-deepseek-v3.2exp-longctx/
```

### Verify Model:
```bash
# Check model files
ls -lh checkpoints/sft-deepseek-v3.2exp-longctx/
```

## ⚙️ Key Configuration (.env)

```bash
# API Key (already set)
EODHD_API_KEY=68f49912abd075.05871806

# Training Dates
TRAIN_END_DATE=2024-12-31    # Train up to this date
TEST_START_DATE=2025-01-01   # Test from this date

# Model
BASE_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
MAX_LENGTH=65536              # Long context window

# Hardware
CPU_ONLY_MODE=true           # Set to false on GPU
```

## 🐛 Troubleshooting

### "ModuleNotFoundError"
```bash
# Activate venv first
source venv/bin/activate
# Then install missing package
pip install [package-name]
```

### "CUDA out of memory"
Edit `configs/sft_config.yaml`:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 32  # Increase this
max_length: 32768  # Or reduce this
```

### "Dataset not found"
```bash
# Make sure you ran these first:
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py
```

### Check Installation:
```bash
python3 << 'EOF'
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
except:
    print("✗ PyTorch not installed")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except:
    print("✗ Transformers not installed")

try:
    import datasets
    print(f"✓ Datasets: {datasets.__version__}")
except:
    print("✗ Datasets not installed")

try:
    import trl
    print(f"✓ TRL: {trl.__version__}")
except:
    print("✗ TRL not installed")
EOF
```

## 📊 Expected Timeline

- **Data prep** (CPU): ~5 minutes for 11k samples
- **Dataset creation** (CPU): ~2 minutes
- **Tokenization** (CPU/GPU): ~5-10 minutes
- **Smoke test** (GPU): ~5-10 minutes (10 steps)
- **Full training** (GPU): ~4-8 hours (depends on GPU, 2 epochs, ~11k samples)
- **Evaluation** (GPU): ~10-30 minutes
- **Backtest** (CPU): ~1-2 minutes

## ✅ Verification Checklist

### Before Training:
- [ ] XML files in `data/raw_xml/`
- [ ] JSONL file created (check with `wc -l data/jsonl/all.jsonl`)
- [ ] HF dataset created (check `data/hf_datasets/sft_dataset/`)
- [ ] `.env` configured correctly
- [ ] venv activated
- [ ] GPU available (run `nvidia-smi`)

### After Training:
- [ ] Model checkpoint exists
- [ ] Manifest.json generated
- [ ] Training logs show decreasing loss
- [ ] No CUDA errors in logs

### After Evaluation:
- [ ] `eval_results.csv` created
- [ ] Classification accuracy > 30% (random is 20%)
- [ ] Financial metrics computed
- [ ] Backtest results reasonable

## 🎯 Success Metrics

**Minimum Viable:**
- Action classification accuracy > 40%
- Hit rate (directional correctness) > 52%
- Model generates valid XML structure
- Training completes without errors

**Good Performance:**
- Action classification accuracy > 60%
- Hit rate > 55%
- Sharpe ratio > 0.5
- Model reasoning is coherent

**Excellent Performance:**
- Action classification accuracy > 75%
- Hit rate > 60%
- Sharpe ratio > 1.0
- Backtested returns beat buy-and-hold

## 📞 Getting Help

1. **Check logs**: `logs/` directory has detailed error messages
2. **Review status**: `IMPLEMENTATION_STATUS.md`
3. **Full docs**: `runbook/README.md`
4. **Test pipeline**: `bash scripts/smoke_test.sh`

## 🎉 You're Ready!

Everything is implemented and tested. Just copy your XML files and run the pipeline!

```bash
# On CPU (now)
python3 src/parsers/xml_to_jsonl.py

# On GPU (later)
bash scripts/run_full_pipeline.sh
```

Good luck! 🚀
