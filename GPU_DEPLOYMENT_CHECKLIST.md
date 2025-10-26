# GPU Server Deployment Checklist

**Purpose:** Systematic checklist for deploying SFT-Charlie on GPU server after CPU validation.

**Context:** Moving from CPU-only testing server to 8×H100 GPU production server.

---

## Pre-Transfer Checklist (CPU Server)

### Code & Configuration
- [ ] All CPU tests passing (see `CPU_TESTING_PLAN.md`)
- [ ] Git repository status clean or all changes committed
- [ ] `.gitignore` updated to exclude large files
- [ ] Configuration files reviewed and finalized:
  - [ ] `.env` (with placeholder API keys)
  - [ ] `configs/sft_config.yaml`
  - [ ] `configs/nemo/finetune.yaml`
  - [ ] `configs/eval_config.yaml`
  - [ ] `configs/backtest_config.yaml`

### Data Preparation
- [ ] XML files processed to JSONL
- [ ] HuggingFace datasets created with time-based splits
- [ ] NeMo JSONL files exported
- [ ] Data statistics documented:
  - [ ] Total records: ___________
  - [ ] Train samples: ___________
  - [ ] Validation samples: ___________
  - [ ] Test samples: ___________
  - [ ] Date range: ___________ to ___________
  - [ ] Total size: ___________ GB

### Data Transfer Options
Choose one:

**Option A: Transfer processed data**
```bash
# Package data for transfer
tar -czf sft_data.tar.gz data/jsonl data/hf_datasets data/nemo

# Size check
ls -lh sft_data.tar.gz
```

**Option B: Regenerate on GPU server**
- [ ] XML files copied to GPU server
- [ ] Processing scripts tested and ready
- [ ] Estimated processing time: ___________ hours

**Option C: Git LFS (if repository has LFS)**
- [ ] Data committed to Git LFS
- [ ] LFS bandwidth sufficient for transfer

### Documentation
- [ ] `CPU_TEST_RESULTS.md` completed
- [ ] Known issues documented
- [ ] Dependencies list finalized
- [ ] This checklist reviewed

---

## GPU Server Requirements

### Hardware Verification
- [ ] **GPUs:** 8×H100 80GB or equivalent
  ```bash
  nvidia-smi --query-gpu=name,memory.total --format=csv
  # Expected: 8 GPUs, each with ~80GB memory
  ```
- [ ] **CPU:** 64+ cores recommended
  ```bash
  lscpu | grep "^CPU(s):"
  ```
- [ ] **RAM:** 512GB+ recommended
  ```bash
  free -h
  ```
- [ ] **Disk:** 500GB+ free space
  ```bash
  df -h /
  ```
- [ ] **NVLink:** Verify GPU interconnect
  ```bash
  nvidia-smi nvlink --status
  # Should show NVLink connections between GPUs
  ```

### Software Requirements
- [ ] **OS:** Linux (Ubuntu 20.04/22.04 or RHEL 8+)
  ```bash
  cat /etc/os-release
  ```
- [ ] **CUDA:** 12.1+ installed
  ```bash
  nvcc --version
  nvidia-smi  # Check driver version
  ```
- [ ] **NVIDIA Drivers:** 530+ recommended
  ```bash
  nvidia-smi | grep "Driver Version"
  ```
- [ ] **Python:** 3.10 or 3.11
  ```bash
  python3 --version
  ```
- [ ] **Git:** 2.0+ with LFS support
  ```bash
  git --version
  git lfs version
  ```

### Network & Access
- [ ] **Internet:** Access to HuggingFace, PyPI
  ```bash
  curl -I https://huggingface.co
  curl -I https://pypi.org
  ```
- [ ] **HuggingFace:** Account and token ready
  ```bash
  # Will need: huggingface-cli login
  ```
- [ ] **Bandwidth:** Sufficient for 40GB+ model download
- [ ] **Firewall:** Ports open for distributed training (if multi-node)

---

## Installation Steps (GPU Server)

### Step 1: Clone Repository
```bash
# SSH into GPU server
ssh user@gpu-server

# Clone repository
cd /workspace  # or your preferred location
git clone <your-repo-url> SFT-Charlie
cd SFT-Charlie

# Verify branch
git branch
git status
```

**Verification:**
- [ ] Repository cloned successfully
- [ ] On correct branch: ___________
- [ ] Latest commit matches CPU server: ___________

---

### Step 2: Environment Setup
```bash
# Install with GPU support and NeMo
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh

# Activate environment
source venv/bin/activate

# Verify installation
python3 << 'EOF'
import torch
import nemo
import megatron

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")
print(f"✓ GPU Count: {torch.cuda.device_count()}")
print(f"✓ NeMo: {nemo.__version__}")
print(f"✓ Megatron: {megatron.__version__}")

# Check each GPU
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
EOF
```

**Verification:**
- [ ] Virtual environment created
- [ ] PyTorch with CUDA support installed
- [ ] NeMo toolkit installed
- [ ] All 8 GPUs detected
- [ ] Each GPU shows ~80GB memory

---

### Step 3: Configure Environment
```bash
# Copy .env template
cp .env.example .env  # or transfer from CPU server

# Edit .env with real credentials
nano .env
```

**Update these variables:**
```bash
# API Keys
EODHD_API_KEY=<your-real-api-key>

# Date Ranges (verify these match your data)
TRAIN_START_DATE=2023-10-24
TRAIN_END_DATE=2024-12-31
VALIDATION_DAYS=30
TEST_START_DATE=2025-01-01

# Training Backend
TRAIN_BACKEND=nemo

# GPU Settings
CPU_ONLY_MODE=false
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: Weights & Biases
WANDB_API_KEY=<optional>
WANDB_PROJECT=sft-charlie
```

**Verification:**
- [ ] `.env` file created with real values
- [ ] API key valid (test with price data script)
- [ ] Dates match processed data

---

### Step 4: Transfer Data

**Option A: Transfer pre-processed data**
```bash
# On CPU server
rsync -avz --progress data/ user@gpu-server:/workspace/SFT-Charlie/data/

# On GPU server, verify
du -sh data/*
ls -la data/jsonl/
ls -la data/hf_datasets/
ls -la data/nemo/
```

**Option B: Regenerate data on GPU server**
```bash
# Copy XML files
rsync -avz --progress data/raw_xml/ user@gpu-server:/workspace/SFT-Charlie/data/raw_xml/

# On GPU server, run pipeline
python3 src/parsers/xml_to_jsonl.py
python3 src/data/convert_dataset.py
python3 src/data/export_nemo_dataset.py \
    --dataset_dir data/hf_datasets/sft_dataset \
    --output_dir data/nemo/sft_dataset
```

**Verification:**
- [ ] Data directories present: `data/jsonl/`, `data/hf_datasets/`, `data/nemo/`
- [ ] Record counts match CPU server
- [ ] File sizes reasonable
- [ ] Spot-check data quality

---

### Step 5: Download & Convert DeepSeek-V3

**This is the most time/storage intensive step.**

#### 5a. Download FP8 Model (~40GB)
```bash
# Create directories
mkdir -p checkpoints/source

# Download with Git LFS
cd checkpoints/source
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base deepseek-v3
cd ../..

# Verify download
du -sh checkpoints/source/deepseek-v3
ls -lh checkpoints/source/deepseek-v3/*.safetensors
```

**Verification:**
- [ ] Model downloaded (~40GB)
- [ ] All `.safetensors` files present
- [ ] `config.json` present
- [ ] No corrupted files

**Estimated Time:** 10-30 minutes (depending on bandwidth)

---

#### 5b. Convert FP8 to BF16 (~40GB)
```bash
# Convert using Triton kernels
bash scripts/convert/convert_deepseek_v3.sh \
    --source checkpoints/source/deepseek-v3 \
    --output checkpoints/bf16/deepseek-v3

# Verify conversion
ls -lh checkpoints/bf16/deepseek-v3/*.safetensors
```

**Verification:**
- [ ] Conversion completed without errors
- [ ] BF16 checkpoint created (~40GB)
- [ ] All layers converted
- [ ] Logs show no warnings

**Estimated Time:** 15-45 minutes (GPU-accelerated)

---

#### 5c. Import to NeMo Format (~20GB)
```bash
# Import to .nemo archive
python3 scripts/convert/import_to_nemo.py \
    --bf16-dir checkpoints/bf16/deepseek-v3 \
    --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1

# Verify .nemo file
ls -lh checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo
```

**Verification:**
- [ ] `.nemo` archive created (~20GB)
- [ ] Import logs show success
- [ ] File size reasonable

**Estimated Time:** 10-30 minutes

**Total Model Setup Time:** ~1-2 hours
**Total Disk Usage:** ~100GB (FP8 + BF16 + NeMo; can delete FP8/BF16 after)

---

### Step 6: Verify NeMo Configuration
```bash
# Check config syntax
python3 -c "
import yaml
with open('configs/nemo/finetune.yaml') as f:
    cfg = yaml.safe_load(f)
print('NeMo Config:')
print(yaml.dump(cfg, default_flow_style=False))
"

# Verify paths in config
python3 << 'EOF'
import yaml
from pathlib import Path

with open('configs/nemo/finetune.yaml') as f:
    cfg = yaml.safe_load(f)

# Check resume path
resume_path = cfg.get('recipe', {}).get('resume_path')
if resume_path and Path(resume_path).exists():
    print(f"✓ Resume path exists: {resume_path}")
else:
    print(f"✗ Resume path missing: {resume_path}")

# Check data paths
data_paths = cfg.get('data', {}).get('paths', [])
for path in data_paths:
    if Path(path).exists():
        print(f"✓ Data path exists: {path}")
    else:
        print(f"✗ Data path missing: {path}")
EOF
```

**Verification:**
- [ ] Config file valid YAML
- [ ] Resume path points to `.nemo` file
- [ ] Data paths point to NeMo JSONL files
- [ ] Hyperparameters reviewed:
  - [ ] `tensor_parallel_size: 8`
  - [ ] `pipeline_parallel_size: 1`
  - [ ] `micro_batch_size: 1`
  - [ ] `global_batch_size: 128`
  - [ ] `seq_length: 65536`

---

## GPU Smoke Tests

### Test 1: GPU Availability
```bash
python3 << 'EOF'
import torch
import nemo

assert torch.cuda.is_available(), "CUDA not available"
assert torch.cuda.device_count() == 8, f"Expected 8 GPUs, got {torch.cuda.device_count()}"

print("✓ CUDA available")
print(f"✓ {torch.cuda.device_count()} GPUs detected")

# Test allocation on each GPU
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        x = torch.randn(1000, 1000, device=f"cuda:{i}")
        y = x @ x.T
        print(f"✓ GPU {i} compute OK")

print("\n✅ All GPUs functional")
EOF
```

**Expected:** All 8 GPUs pass compute test

---

### Test 2: NeMo Model Loading
```bash
# Test loading .nemo checkpoint
python3 << 'EOF'
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
import torch

checkpoint_path = "checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo"

print(f"Loading checkpoint: {checkpoint_path}")
print("This may take 2-5 minutes...")

try:
    # Load with distributed settings
    model = MegatronGPTModel.restore_from(
        checkpoint_path,
        trainer=None,  # Will create default trainer
        override_config_path=None
    )
    print("✓ Model loaded successfully")
    print(f"✓ Model type: {type(model)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
EOF
```

**Expected:** Model loads without errors (may take 2-5 minutes)

**If this fails:** Check tensor/pipeline parallel settings match `.nemo` file

---

### Test 3: Minimal Training Test
```bash
# Run training for 10 steps only
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/smoke_test \
    --max_steps 10 \
    --log_every_n_steps 1

# Check outputs
ls -la checkpoints/nemo_runs/smoke_test/
cat logs/train_nemo_*.log | tail -50
```

**Expected:**
- Training starts without errors
- All 8 GPUs utilized
- Loss computed for 10 steps
- Checkpoint saved

**Estimated Time:** 5-15 minutes

**Verification:**
- [ ] Training initialized
- [ ] Distributed setup successful (8 GPUs)
- [ ] Loss decreasing (or stable)
- [ ] No OOM errors
- [ ] Checkpoint saved

---

## Full Training Execution

### Pre-Training Checklist
- [ ] All smoke tests passed
- [ ] Disk space sufficient (100GB+ free)
- [ ] Monitoring setup (WandB/TensorBoard)
- [ ] Training config reviewed
- [ ] Estimated time calculated: ___________ hours
  - Formula: `(num_samples × epochs × seq_length) / (global_batch_size × throughput_tokens_per_sec)`
  - Example: `(10000 × 2 × 65536) / (128 × 50000) ≈ 200 hours ≈ 8 days`

### Launch Training
```bash
# Full training run
nohup python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/main \
    > logs/training_main.log 2>&1 &

# Monitor
tail -f logs/training_main.log

# Or with screen/tmux
screen -S training
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/main

# Detach: Ctrl+A, D
# Reattach: screen -r training
```

### Monitoring During Training
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f logs/training_main.log

# WandB (if configured)
# Visit: https://wandb.ai/<your-project>

# TensorBoard (if configured)
tensorboard --logdir checkpoints/nemo_runs/main --port 6006
```

**Monitor These Metrics:**
- [ ] GPU utilization: ~90-100%
- [ ] GPU memory usage: ~70-80GB per GPU
- [ ] Training loss: Decreasing
- [ ] Learning rate: Following schedule
- [ ] Throughput: ___________ tokens/sec
- [ ] ETA: ___________ hours remaining

---

## Post-Training Validation

### Step 1: Checkpoint Verification
```bash
# Check final checkpoint
ls -lh checkpoints/nemo_runs/main/*.nemo

# Verify checkpoint integrity
python3 << 'EOF'
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

checkpoint = "checkpoints/nemo_runs/main/final.nemo"
print(f"Loading checkpoint: {checkpoint}")

try:
    model = MegatronGPTModel.restore_from(checkpoint)
    print("✓ Checkpoint valid and loadable")
except Exception as e:
    print(f"✗ Checkpoint corrupt: {e}")
EOF
```

**Verification:**
- [ ] Final checkpoint exists
- [ ] Size reasonable (~20GB full / ~500MB LoRA)
- [ ] Loads without errors

---

### Step 2: Evaluation
```bash
# Run evaluation on test set
python3 src/eval/evaluate_nemo.py \
    --model checkpoints/nemo_runs/main/final.nemo \
    --dataset data/nemo/sft_dataset \
    --split test \
    --results results/eval_results.csv \
    --output_predictions results/predictions.jsonl

# Check results
head -20 results/eval_results.csv
python3 -c "
import pandas as pd
df = pd.read_csv('results/eval_results.csv')
print(f'Evaluated {len(df)} samples')
print(df.describe())
"
```

**Estimated Time:** 1-4 hours (depending on test set size)

**Verification:**
- [ ] Evaluation completes without errors
- [ ] Results CSV created
- [ ] Predictions JSONL created
- [ ] Metrics computed:
  - [ ] Action accuracy: ___________
  - [ ] Precision: ___________
  - [ ] Recall: ___________
  - [ ] F1 score: ___________

---

### Step 3: Backtesting
```bash
# Run portfolio backtest
python3 src/backtest/trading_backtest.py \
    --eval_jsonl results/eval_results.csv \
    --config configs/backtest_config.yaml \
    --out backtests/main_run.csv

# View results
cat backtests/main_run.csv
```

**Verification:**
- [ ] Backtest completes
- [ ] Financial metrics computed:
  - [ ] Hit rate: ___________
  - [ ] Mean return: ___________
  - [ ] Sharpe ratio: ___________
  - [ ] Max drawdown: ___________
- [ ] Results make sense (no extreme outliers)

---

## Performance Expectations

### Training
- **Throughput:** 50,000-100,000 tokens/sec (8×H100)
- **GPU Utilization:** 85-100%
- **Memory per GPU:** 70-80GB / 80GB
- **Time per epoch:** Depends on dataset size
  - ~10K samples: 1-2 days
  - ~100K samples: 1-2 weeks

### Evaluation
- **Throughput:** 10,000-50,000 tokens/sec
- **Time per sample:** 1-5 seconds
- **Test set (1000 samples):** 1-2 hours

### Expected Metrics
- **Action Accuracy:** 60-75%
- **Hit Rate:** 55-65%
- **Sharpe Ratio:** 0.5-1.5
- **Max Drawdown:** -10% to -30%

---

## Troubleshooting Guide

### Issue: CUDA Out of Memory (OOM)
**Symptoms:** Training crashes with OOM error

**Solutions:**
1. Reduce `micro_batch_size` in config (try 1 → 0.5)
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Reduce `seq_length`: `65536 → 32768`
4. Use LoRA instead of full fine-tuning: `peft: lora`

---

### Issue: Slow Training Throughput
**Symptoms:** <30,000 tokens/sec

**Solutions:**
1. Check GPU utilization: `nvidia-smi`
2. Enable flash attention: `flash_attention: true`
3. Optimize dataloader: `num_workers: 8`
4. Check I/O bottleneck: Monitor disk usage
5. Verify NVLink active: `nvidia-smi nvlink --status`

---

### Issue: Loss Not Decreasing
**Symptoms:** Loss plateaus or increases

**Solutions:**
1. Check learning rate: May need adjustment
2. Verify data quality: Inspect training samples
3. Increase batch size: `global_batch_size: 256`
4. Reduce learning rate: `5e-6 → 1e-6`
5. Check for data leakage: Validate time splits

---

### Issue: Model Produces Gibberish
**Symptoms:** Evaluation shows random outputs

**Solutions:**
1. Verify checkpoint loaded correctly
2. Check tokenizer matches training tokenizer
3. Ensure special tokens added
4. Review prompt template in evaluation
5. Check temperature/sampling settings

---

### Issue: Evaluation Errors
**Symptoms:** evaluate_nemo.py crashes

**Solutions:**
1. Verify test data format matches training
2. Check NeMo JSONL structure
3. Ensure model and data compatible
4. Test with small subset first: `--max_samples 10`

---

## Data Management

### Disk Usage Monitoring
```bash
# Check disk usage
df -h

# Breakdown by directory
du -sh checkpoints/* data/* logs/* results/*

# Cleanup old runs
rm -rf checkpoints/nemo_runs/smoke_test
rm -rf logs/old_*.log
```

### Backup Strategy
```bash
# Backup trained checkpoint
rsync -avz checkpoints/nemo_runs/main/ \
    backup-server:/backups/sft-charlie/$(date +%Y%m%d)/

# Backup critical results
tar -czf results_$(date +%Y%m%d).tar.gz results/ backtests/
```

---

## Post-Deployment Checklist

### Verification
- [ ] Training completed successfully
- [ ] Final checkpoint saved and verified
- [ ] Evaluation metrics computed
- [ ] Backtest results generated
- [ ] Results match expectations
- [ ] No critical errors in logs

### Documentation
- [ ] Training manifest saved
- [ ] Metrics documented
- [ ] Config files backed up
- [ ] Known issues documented
- [ ] Results shared with team

### Optimization (Optional)
- [ ] Hyperparameter tuning experiments planned
- [ ] Alternative configurations tested
- [ ] Performance benchmarks recorded
- [ ] Next iteration planned

---

## Quick Reference Commands

### Check System
```bash
# GPUs
nvidia-smi

# Disk
df -h

# Environment
conda env list  # or: source venv/bin/activate

# Git status
git status
git log -1
```

### Training
```bash
# Start
python3 src/train/train_nemo.py --config configs/nemo/finetune.yaml

# Monitor
tail -f logs/training_main.log
watch -n 1 nvidia-smi

# Resume (if interrupted)
python3 src/train/train_nemo.py --config configs/nemo/finetune.yaml --resume
```

### Evaluation
```bash
# Evaluate
python3 src/eval/evaluate_nemo.py --model <checkpoint> --dataset <data>

# Backtest
python3 src/backtest/trading_backtest.py --eval_jsonl results/eval_results.csv
```

---

## Contact & Support

**Issues:** Check `logs/` directory and search error messages
**Documentation:** See `README.md`, `QUICK_START.md`, `runbook/README.md`
**NeMo Docs:** https://docs.nvidia.com/nemo-framework/

---

## Final Pre-Launch Checklist

Before starting training on GPU server:

- [ ] All CPU tests passed
- [ ] All GPU smoke tests passed
- [ ] Monitoring configured
- [ ] Backup strategy in place
- [ ] Estimated time/cost calculated
- [ ] Team notified of training start
- [ ] Emergency stop procedure documented

**Ready to deploy?** YES / NO

**Deployment Date:** ___________

**Expected Completion:** ___________
