# NeMo Migration Guide

## Overview

This project has been migrated from Hugging Face TRL/SFTTrainer to NVIDIA NeMo for DeepSeek-V3 fine-tuning. This migration enables:

- **Native DeepSeek-V3 support** with proper FP8→BF16 conversion
- **Full-parameter fine-tuning** on 8×H100 topology (not just LoRA)
- **Better multi-GPU scaling** via Megatron-Core parallelism strategies
- **Long-context optimization** supporting up to 131k tokens
- **Production-grade training** with NeMo's battle-tested infrastructure

## What Changed

### Training Framework
- **Before:** TRL `SFTTrainer` with PEFT/LoRA + QLoRA (4-bit quantization)
- **After:** NeMo `finetune_recipe` with full-parameter or optional LoRA

### Model Format
- **Before:** Hugging Face checkpoint (adapter weights + tokenizer)
- **After:** NeMo `.nemo` archive (includes model, tokenizer, config, parallelism metadata)

### Data Format
- **Before:** Tokenized HF Arrow shards
- **After:** NeMo JSONL (`training.jsonl`, `validation.jsonl`, `test.jsonl`)

### Evaluation
- **Before:** `evaluate_sft.py` loads HF checkpoint via `AutoModelForCausalLM`
- **After:** `evaluate_nemo.py` restores `.nemo` via `llm.restore_model`

### Parallelism
- **Before:** Single-GPU or DeepSpeed ZeRO-3
- **After:** Tensor parallel (TP=8), pipeline parallel (PP=1-8), sequence parallel

## Migration Steps (Already Complete)

1. ✅ Vendored FP8→BF16 conversion scripts from DeepSeek
2. ✅ Created NeMo import tooling (`scripts/convert/`)
3. ✅ Built NeMo training entrypoint (`src/train/train_nemo.py`)
4. ✅ Created NeMo config system (`configs/nemo/finetune.yaml`)
5. ✅ Added dataset export (`src/data/export_nemo_dataset.py`)
6. ✅ Rewrote evaluation (`src/eval/evaluate_nemo.py`)
7. ✅ Updated backtesting for NeMo outputs
8. ✅ Updated all documentation and scripts

## New Workflow

### Step 1: Model Conversion (One-Time)

```bash
# Download FP8 checkpoint
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base checkpoints/source/deepseek-v3

# Convert FP8→BF16
bash scripts/convert/convert_deepseek_v3.sh \
  --source checkpoints/source/deepseek-v3 \
  --output checkpoints/bf16/deepseek-v3

# Import to NeMo
python3 scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo \
  --tensor-parallel 8
```

**Time:** 2-4 hours (download + conversion)
**Disk:** ~60GB temporary, ~20GB final

### Step 2: Data Preparation

```bash
# Parse XML
python3 src/parsers/xml_to_jsonl.py \
  --input_dir data/raw_xml \
  --output_file data/jsonl/all.jsonl

# Create HF dataset (intermediate)
python3 src/data/convert_dataset.py \
  --jsonl data/jsonl/all.jsonl \
  --out_dir data/hf_datasets/sft_dataset

# Export NeMo JSONL
python3 src/data/export_nemo_dataset.py \
  --dataset_dir data/hf_datasets/sft_dataset \
  --output_dir data/nemo/sft_dataset \
  --template chatml \
  --include_metadata
```

**Output:** `data/nemo/sft_dataset/{training,validation,test}.jsonl`

### Step 3: Training

```bash
# Update config
nano configs/nemo/finetune.yaml

# Run training
python3 src/train/train_nemo.py \
  --config configs/nemo/finetune.yaml \
  --output checkpoints/nemo_runs/main
```

**Time:** 4-24 hours depending on dataset size
**Output:** `checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo`

### Step 4: Evaluation

```bash
python3 src/eval/evaluate_nemo.py \
  --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo \
  --dataset data/nemo/sft_dataset \
  --split test \
  --results results/eval_results.csv
```

### Step 5: Backtesting

```bash
python3 src/backtest/trading_backtest.py \
  --eval_jsonl results/eval_results.csv \
  --config configs/backtest_config.yaml \
  --out backtests/baseline.csv
```

## Configuration Reference

### NeMo Config Structure

```yaml
recipe:
  factory: deepseek_v3               # NeMo recipe name
  name: deepseek_v3_finetune         # Run name
  resume_path: <path-to-.nemo>       # Base checkpoint

dataset:
  path: data/nemo/sft_dataset        # JSONL directory
  template: chatml                   # Prompt template
  label_key: output                  # Response field name
  answer_only_loss: true             # Mask prompt in loss

train:
  peft: none                         # 'none', 'lora', 'dora'
  seq_length: 65536                  # Max sequence length
  micro_batch_size: 1                # Per-device batch
  global_batch_size: 128             # Total across devices
  num_nodes: 1                       # Cluster nodes
  gpus_per_node: 8                   # GPUs per node
  performance_mode: false            # Enable optimizations
```

### Parallelism Settings (Auto-configured)

For **full fine-tune** (`peft: none`):
- Tensor parallel: 1
- Pipeline parallel: 8
- Expert parallel: 64
- Sequence parallel: False

For **LoRA** (`peft: lora`):
- Tensor parallel: 8
- Pipeline parallel: 5
- Expert parallel: 1
- Sequence parallel: True

## Legacy Scripts (Deprecated)

The following files are retained for reference but should not be used:

- ❌ `src/train/train_sft.py` → Use `src/train/train_nemo.py`
- ❌ `src/data/tokenize_and_shard.py` → Use `src/data/export_nemo_dataset.py`
- ❌ `src/eval/evaluate_sft.py` → Use `src/eval/evaluate_nemo.py`
- ❌ `configs/sft_config.yaml` → Use `configs/nemo/finetune.yaml`
- ❌ `configs/deepspeed_stage3.json` → NeMo handles parallelism internally

## Hardware Requirements

### Development
- 1×A100/H100 for smoke tests
- 100 GB disk space

### Production
- 8×H100 80GB (NVLink)
- 512 GB RAM
- 500 GB NVMe storage
- High-bandwidth network (InfiniBand/RoCE)

## Troubleshooting

### "NeMo toolkit is required"
```bash
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

### "Failed to load .nemo checkpoint"
Ensure model was imported with correct tensor/pipeline parallel settings matching your training config.

### "CUDA out of memory"
- Reduce `seq_length` in config
- Enable `performance_mode: true`
- Switch to LoRA: `peft: lora`

### "Training hangs at initialization"
Check that `num_nodes` and `gpus_per_node` match your actual cluster configuration.

## FAQ

**Q: Can I still use the old TRL scripts?**
A: They're retained but unsupported. Set `TRAIN_BACKEND=trl` in environment to force legacy path.

**Q: Do I need to re-convert the model for each run?**
A: No, conversion is one-time. The `.nemo` archive is reused across training runs.

**Q: Can I export NeMo checkpoints back to Hugging Face?**
A: Yes, use NeMo's export utilities:
```bash
python -m nemo.collections.llm.export.export_lora_to_hf \
  --nemo-file <path>.nemo \
  --output-dir checkpoints/hf_export
```

**Q: What happened to QLoRA/4-bit quantization?**
A: NeMo focuses on full-precision or LoRA. For memory constraints, use LoRA instead of QLoRA.

**Q: How do I resume training?**
A: Pass `--resume <checkpoint>.nemo` to `train_nemo.py`.

## Support

For NeMo-specific issues, consult:
- [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/)
- [NeMo GitHub](https://github.com/NVIDIA/NeMo)

For pipeline issues, check `logs/` directory and verify data integrity with validation scripts.
