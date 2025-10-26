# Changelog

## [2.0.0] - 2025-10-26 - NeMo Migration

### Major Changes

**Migration from TRL/Hugging Face to NVIDIA NeMo**

This release represents a complete framework migration from TRL/SFTTrainer to NVIDIA NeMo for production-grade DeepSeek-V3 fine-tuning.

### Added

- **Model Conversion Pipeline**
  - `scripts/convert/fp8_cast_bf16.py` - FP8→BF16 conversion with Triton kernels
  - `scripts/convert/kernel.py` - DeepSeek dequantization kernels
  - `scripts/convert/convert_deepseek_v3.sh` - Wrapper script for conversion
  - `scripts/convert/import_to_nemo.py` - NeMo archive import utility

- **NeMo Training**
  - `src/train/train_nemo.py` - NeMo-based training entrypoint
  - `configs/nemo/finetune.yaml` - NeMo training configuration
  - `requirements_nemo.txt` - NeMo-specific dependencies

- **NeMo Data Pipeline**
  - `src/data/export_nemo_dataset.py` - HF→NeMo JSONL export
  - Support for `training.jsonl`, `validation.jsonl`, `test.jsonl` format
  - Tokenizer extension with special tokens

- **NeMo Evaluation**
  - `src/eval/evaluate_nemo.py` - NeMo checkpoint evaluation
  - `src/eval/metrics.py` - Refactored metrics computation
  - `configs/eval_config.json` - Metrics configuration

- **Documentation**
  - `README.md` - Comprehensive project overview
  - `NEMO_MIGRATION.md` - Detailed migration guide
  - `PROJECT_STATUS.md` - Current implementation status
  - `runbook/README.md` - Complete rewrite for NeMo

### Changed

- **Training Backend**
  - Default training now uses NeMo (`TRAIN_BACKEND=nemo`)
  - Full-parameter fine-tuning support (not limited to LoRA)
  - 8×H100 topology with tensor parallelism

- **Model Format**
  - Output changed from HF checkpoint to `.nemo` archive
  - Includes model, tokenizer, config, parallelism metadata
  - Size: ~20GB for full model, ~500MB for LoRA

- **Scripts**
  - `scripts/run_full_pipeline.sh` - Updated for NeMo workflow
  - `scripts/setup_env.sh` - Added NeMo installation support
  - Evaluation and backtest updated for JSONL outputs

- **Configuration**
  - Moved from YAML-only to Hydra/OmegaConf style
  - Simplified config structure for NeMo recipes
  - Environment variable overrides retained

### Deprecated

- `src/train/train_sft.py` - Use `src/train/train_nemo.py`
- `src/eval/evaluate_sft.py` - Use `src/eval/evaluate_nemo.py`
- `src/data/tokenize_and_shard.py` - Use `src/data/export_nemo_dataset.py`
- `configs/sft_config.yaml` - Use `configs/nemo/finetune.yaml`
- `configs/deepspeed_stage3.json` - NeMo handles parallelism internally

### Technical Details

#### Parallelism Changes

**Full Fine-Tune:**
- Tensor Parallel: 1
- Pipeline Parallel: 8
- Expert Parallel: 64

**LoRA Fine-Tune:**
- Tensor Parallel: 8
- Pipeline Parallel: 5
- Expert Parallel: 1

#### Breaking Changes

- `.nemo` checkpoints not compatible with HF `AutoModelForCausalLM`
- Training requires NeMo toolkit installation
- Minimum 8×H100 for full fine-tune (5×H100 for LoRA)
- FP8→BF16 conversion required as one-time setup

#### Migration Path

1. Convert DeepSeek-V3 FP8→BF16→.nemo
2. Export HF datasets to NeMo JSONL
3. Update training to use `train_nemo.py`
4. Update evaluation to use `evaluate_nemo.py`
5. Verify backtest compatibility

### Removed

- `IMPLEMENTATION_STATUS.md` - Replaced by `PROJECT_STATUS.md`
- `PROJECT_SUMMARY.txt` - Replaced by `README.md`
- Direct TRL dependencies in default workflow

### Fixed

- Long-context support now properly handles 131k tokens
- Distributed training more stable with NeMo's Megatron backend
- Memory efficiency improved with proper parallelism strategies

---

## [1.0.0] - 2025-10-26 - Initial TRL Implementation

### Added

- Complete TRL/Hugging Face pipeline
- XML parsing and data validation
- Time-based dataset splitting
- LoRA/QLoRA training with bitsandbytes
- Financial evaluation and backtesting
- Price data integration (eodhd + yfinance)
- Comprehensive logging and manifests

### Features

- DeepSeek-V3.2-Exp support via Hugging Face
- 65k context window with gradient checkpointing
- 4-bit quantization (QLoRA) for memory efficiency
- Special token handling for structured output
- Dual-source price data with fallback
- Reproducibility tracking

---

## Migration Timeline

- **October 26, 2025:** NeMo migration initiated
- **October 26, 2025:** Conversion pipeline implemented
- **October 26, 2025:** Training and evaluation migrated
- **October 26, 2025:** Documentation updated
- **October 26, 2025:** Migration complete

## Upgrade Instructions

### From 1.x to 2.x

**Prerequisites:**
- 8×H100 GPU cluster
- CUDA 12.1+
- 500 GB disk space

**Steps:**

1. Install NeMo dependencies:
```bash
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

2. Convert existing DeepSeek model:
```bash
bash scripts/convert/convert_deepseek_v3.sh \
  --source checkpoints/source/deepseek-v3 \
  --output checkpoints/bf16/deepseek-v3

python3 scripts/convert/import_to_nemo.py \
  --bf16-dir checkpoints/bf16/deepseek-v3 \
  --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo
```

3. Export datasets to NeMo format:
```bash
python3 src/data/export_nemo_dataset.py \
  --dataset_dir data/hf_datasets/sft_dataset \
  --output_dir data/nemo/sft_dataset
```

4. Update training command:
```bash
# Old (v1.x):
python3 src/train/train_sft.py --config configs/sft_config.yaml

# New (v2.x):
python3 src/train/train_nemo.py --config configs/nemo/finetune.yaml --output checkpoints/nemo_runs/main
```

5. Update evaluation command:
```bash
# Old (v1.x):
python3 src/eval/evaluate_sft.py --model_dir checkpoints/sft-run

# New (v2.x):
python3 src/eval/evaluate_nemo.py --model checkpoints/nemo_runs/main/deepseek_v3_finetune.nemo
```

**Note:** Existing TRL checkpoints cannot be directly converted to NeMo format. You must retrain from the base model.

---

## Version Compatibility

| Version | Framework | DeepSeek Support | Min Hardware |
|---------|-----------|------------------|--------------|
| 1.x | TRL/HF | V3.2-Exp | 1×A100 40GB |
| 2.x | NeMo | V3-Base | 8×H100 80GB |

---

For detailed migration information, see [NEMO_MIGRATION.md](NEMO_MIGRATION.md).
