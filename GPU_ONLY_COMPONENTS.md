# GPU-Only Components - Cannot Test on CPU Server

This document details components that **CANNOT** be tested on the CPU-only server and **MUST** be validated on the GPU server.

---

## Executive Summary

**~20% of the pipeline requires GPU hardware** and cannot be validated on CPU-only servers. These components involve:
1. Large model operations (DeepSeek-V3 with 685B parameters)
2. GPU-specific operations (CUDA kernels, Triton)
3. Distributed training across multiple GPUs
4. Model inference at scale

**Impact:** These components represent the most expensive and time-critical parts of the pipeline. Thorough CPU testing minimizes GPU server costs by ensuring everything else works before deployment.

---

## 1. Model Conversion (FP8 → BF16)

### Component
**File:** `scripts/convert/fp8_cast_bf16.py`

### Why GPU Required
- Uses **Triton GPU kernels** for efficient FP8→BF16 casting
- Model size (40GB+) requires GPU memory for fast processing
- CUDA operations throughout conversion process

### Technical Details
```python
# From fp8_cast_bf16.py
@triton.jit
def fp8_to_bf16_kernel(...)
    # Triton kernel - GPU only
```

Triton is a GPU-only language that compiles to CUDA. No CPU fallback exists.

### Workaround
**None.** This step is fundamentally GPU-dependent.

### GPU Server Requirements
- **GPU:** 1× H100 80GB minimum (can use single GPU)
- **Memory:** 80GB+ GPU memory
- **Time:** 15-45 minutes
- **Disk:** 80GB (40GB input + 40GB output)

### Test Command (GPU Only)
```bash
bash scripts/convert/convert_deepseek_v3.sh \
    --source checkpoints/source/deepseek-v3 \
    --output checkpoints/bf16/deepseek-v3
```

### Expected Output
- BF16 safetensors files in output directory
- Conversion logs showing layer-by-layer progress
- No NaN or Inf values in converted weights

### Validation
```bash
# Verify output files exist
ls -lh checkpoints/bf16/deepseek-v3/*.safetensors

# Check for errors in logs
grep -i "error\|nan\|inf" logs/conversion.log
```

---

## 2. NeMo Model Import

### Component
**File:** `scripts/convert/import_to_nemo.py`

### Why GPU Required
- NeMo framework requires **CUDA-enabled PyTorch**
- Model parallelism setup requires GPU context
- Tensor/pipeline parallel configuration needs actual GPUs

### Technical Details
```python
# Requires CUDA PyTorch
import nemo
from megatron.core import parallel_state

# Initialize parallel state (needs GPUs)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,  # Needs 8 GPUs
    pipeline_model_parallel_size=1
)
```

### Workaround
**Limited:** Could theoretically run with `CUDA_VISIBLE_DEVICES=""` but would fail at parallelism initialization.

### GPU Server Requirements
- **GPU:** 8× H100 80GB (for TP=8 configuration)
- **Memory:** 40GB+ per GPU during import
- **Time:** 10-30 minutes
- **Disk:** 20GB (.nemo output)

### Test Command (GPU Only)
```bash
python3 scripts/convert/import_to_nemo.py \
    --bf16-dir checkpoints/bf16/deepseek-v3 \
    --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1
```

### Expected Output
- `.nemo` archive file (~20GB)
- Import logs showing successful sharding across 8 GPUs
- Model configuration saved

### Validation
```bash
# Verify .nemo file
ls -lh checkpoints/nemo/*.nemo

# Test loading (requires GPU)
python3 << 'EOF'
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

model = MegatronGPTModel.restore_from(
    "checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo",
    trainer=None
)
print(f"✓ Model loaded: {type(model)}")
EOF
```

---

## 3. NeMo Training

### Component
**File:** `src/train/train_nemo.py`

### Why GPU Required
- Model too large for CPU memory (685B parameters)
- Training loop uses **CUDA operations** extensively
- Distributed training requires **multiple GPUs** with NVLink
- Mixed precision training (BF16/FP32) requires GPU support

### Technical Details
```python
# From train_nemo.py
trainer = nl.Trainer(
    devices=8,  # Number of GPUs
    num_nodes=1,
    strategy=nl.MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1
    )
)
```

Cannot simulate 8-GPU distributed training on CPU.

### Workaround
**None** for actual training. CPU testing covers:
- Configuration parsing
- Data loading logic
- Manifest creation
- Logging setup

But **not**:
- Actual model forward/backward pass
- Gradient computation
- Optimizer steps
- Checkpoint saving with real model

### GPU Server Requirements
- **GPU:** 8× H100 80GB (mandatory for TP=8)
- **Memory:** 70-80GB per GPU during training
- **Time:** Days to weeks depending on dataset size
- **Disk:** 100GB+ for checkpoints and logs

### Test Command (GPU Only)
```bash
# Smoke test (10 steps)
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/smoke_test \
    --max_steps 10

# Full training
python3 src/train/train_nemo.py \
    --config configs/nemo/finetune.yaml \
    --output checkpoints/nemo_runs/main
```

### Expected Output (Smoke Test)
- Training initializes across 8 GPUs
- Loss computed for 10 steps
- Checkpoint saved
- GPU utilization ~90-100%
- No OOM errors

### Validation
```bash
# Monitor during training
nvidia-smi  # All 8 GPUs should show high utilization
tail -f logs/train_nemo_*.log

# After training
ls -lh checkpoints/nemo_runs/main/*.nemo
```

---

## 4. Model Evaluation (Full Scale)

### Component
**File:** `src/eval/evaluate_nemo.py`

### Why GPU Required
- Model inference requires GPU
- Batch processing of test set (1000s of samples)
- Model loaded in distributed manner across 8 GPUs

### Technical Details
```python
# From evaluate_nemo.py
model = MegatronGPTModel.restore_from(
    checkpoint_path,
    trainer=trainer  # Needs GPU trainer
)

# Inference
predictions = model.generate(
    inputs,
    max_length=2048
)  # Uses GPU
```

### Limited CPU Testing
**Can test on CPU:**
- Data loading from NeMo JSONL
- Prompt formatting
- Output parsing (action extraction)
- Metrics computation logic

**Cannot test on CPU:**
- Actual model inference
- Batch processing with real model
- Generation quality

### GPU Server Requirements
- **GPU:** 8× H100 80GB (same as training)
- **Memory:** 60-70GB per GPU
- **Time:** 1-4 hours for typical test set
- **Disk:** 10GB for results

### Test Command (GPU Only)
```bash
python3 src/eval/evaluate_nemo.py \
    --model checkpoints/nemo_runs/main/final.nemo \
    --dataset data/nemo/sft_dataset \
    --split test \
    --results results/eval_results.csv \
    --output_predictions results/predictions.jsonl
```

### Expected Output
- CSV with predictions for each test sample
- JSONL with full model outputs
- Metrics summary JSON
- Action accuracy, precision, recall, F1

### Validation
```bash
# Check results
head -20 results/eval_results.csv

# Compute metrics
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('results/eval_results.csv')
print(f"Samples evaluated: {len(df)}")
print(f"Action accuracy: {(df['predicted_action'] == df['true_action']).mean():.2%}")
EOF
```

---

## 5. Tokenizer Extension (Full)

### Component
**File:** `src/data/export_nemo_dataset.py` (tokenizer portion)

### Why GPU Required (Partially)
- DeepSeek-V3 tokenizer download is 40GB+ (includes model files)
- Loading tokenizer allocates substantial memory
- While tokenizer itself can run on CPU, it's bundled with model

### Workaround
**Can test tokenizer logic with smaller model:**

```python
# CPU-compatible test with small model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 117M - CPU OK

# Test special token addition
special_tokens = {
    "additional_special_tokens": [
        "<reasoning>", "</reasoning>",
        "<action>", "</action>",
        # ...
    ]
}
tokenizer.add_special_tokens(special_tokens)
```

But **cannot test** with actual DeepSeek-V3 tokenizer without downloading model.

### GPU Server Testing
```bash
python3 src/data/export_nemo_dataset.py \
    --dataset_dir data/hf_datasets/sft_dataset \
    --output_dir data/nemo/sft_dataset \
    --base_model deepseek-ai/DeepSeek-V3-Base
```

### Expected Output
- Extended tokenizer saved
- NeMo JSONL with proper token IDs
- Special tokens correctly added to vocabulary

---

## 6. Performance Benchmarking

### Component
Real-world performance metrics

### Why GPU Required
- Throughput measurements (tokens/second)
- GPU utilization profiling
- Memory usage patterns
- Distributed training efficiency

### Metrics That Need GPU
- **Training Throughput:** tokens/second across 8 GPUs
- **Inference Latency:** time per prediction
- **GPU Utilization:** % usage per GPU
- **Memory Efficiency:** peak memory usage
- **NVLink Bandwidth:** inter-GPU communication speed

### Validation Commands (GPU Only)
```bash
# During training
nvidia-smi dmon -s u -c 60  # Monitor utilization for 60 seconds

# Throughput
grep "tokens/s" logs/train_nemo_*.log

# NVLink stats
nvidia-smi nvlink --status
```

---

## 7. Model Download (Optional on CPU)

### Component
Downloading DeepSeek-V3 base model

### Technically Possible on CPU
Yes, downloading is CPU-compatible, but:

**Considerations:**
- **Size:** 40GB download
- **Time:** 10-30 minutes (bandwidth dependent)
- **Storage:** Need 40GB free space
- **Bandwidth:** May be limited on CPU server

### Decision Matrix

| Scenario | Download on CPU Server? |
|----------|-------------------------|
| Fast CPU server network | ✅ Yes - saves GPU server time/cost |
| Slow CPU server network | ❌ No - download on GPU server |
| Limited CPU disk space | ❌ No - download on GPU server |
| Need to verify model files | ✅ Yes - can verify checksums on CPU |

### Download Command (CPU Compatible)
```bash
# Install Git LFS
git lfs install

# Clone model (CPU OK)
cd checkpoints/source
git lfs clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base deepseek-v3

# Verify (CPU OK)
cd deepseek-v3
ls -lh *.safetensors
md5sum *.safetensors > checksums.txt
```

### Recommendation
**Download on CPU server if:**
- CPU server has faster network
- Want to verify files before GPU deployment
- GPU server hourly cost is high

**Download on GPU server if:**
- CPU server disk space limited
- Network speeds similar
- Want to minimize data transfers

---

## Summary Table

| Component | CPU Test | GPU Test | GPU Required | Priority |
|-----------|----------|----------|--------------|----------|
| **FP8→BF16 Conversion** | ❌ None | ✅ Full | Mandatory | Critical |
| **NeMo Import** | ❌ None | ✅ Full | Mandatory | Critical |
| **Training** | ⚠️ Config only | ✅ Full | Mandatory | Critical |
| **Evaluation** | ⚠️ Logic only | ✅ Full | Mandatory | High |
| **Tokenizer (DeepSeek)** | ⚠️ Small model | ✅ Full | Recommended | Medium |
| **Benchmarking** | ❌ None | ✅ Full | Optional | Low |
| **Model Download** | ✅ Possible | ✅ Possible | Optional | Low |

---

## Testing Strategy

### On CPU Server (Current)
✅ **Focus on these:**
1. Data pipeline (XML → JSONL → HF → NeMo)
2. Configuration validation
3. Utility functions
4. Data quality checks
5. Time-based split validation
6. Environment setup scripts
7. Documentation

⏭️ **Skip these:**
1. Model conversion
2. Training execution
3. Model inference
4. Performance profiling

### On GPU Server (Next)
🎯 **First priority:**
1. Model conversion smoke test (1 layer)
2. NeMo import smoke test
3. Training smoke test (10 steps)
4. Evaluation smoke test (10 samples)

🚀 **Second priority:**
1. Full model conversion
2. Full NeMo import
3. Short training run (100 steps)
4. Partial evaluation (100 samples)

💪 **Production:**
1. Full training (all epochs)
2. Complete evaluation (all test samples)
3. Backtesting
4. Performance optimization

---

## Risk Mitigation

### High-Risk Areas (Test Early on GPU)
1. **Model Conversion:** Could fail silently, producing corrupted weights
   - **Mitigation:** Validate converted weights with checksums
   - **Test:** Convert single layer first

2. **OOM Errors:** GPU memory insufficient for batch size
   - **Mitigation:** Start with micro_batch_size=1
   - **Test:** Smoke test with minimal data

3. **Distributed Setup:** Parallelism misconfigured
   - **Mitigation:** Match TP/PP to .nemo file config
   - **Test:** Verify GPU detection before loading model

4. **Data Format Mismatch:** NeMo expects different format
   - **Mitigation:** Validate JSONL structure
   - **Test:** Load 1 batch before full training

### Medium-Risk Areas (Test During Development)
1. **Slow Throughput:** Inefficient data loading or training
   - **Mitigation:** Monitor GPU utilization
   - **Test:** Compare to expected tokens/second

2. **Poor Model Quality:** Model not learning
   - **Mitigation:** Monitor loss curves
   - **Test:** Evaluate on validation set during training

### Low-Risk Areas (Can Test Late)
1. **Logging Issues:** Logs not saving correctly
   - **Mitigation:** Check logs directory
   - **Test:** Verify logs after first checkpoint

2. **Checkpoint Corruption:** Saved checkpoints unusable
   - **Mitigation:** Test loading checkpoints
   - **Test:** Resume from checkpoint

---

## Estimated GPU Server Time

### One-Time Setup (~2-4 hours)
- Model download: 15-30 min
- FP8→BF16 conversion: 30-60 min
- NeMo import: 15-30 min
- Environment setup: 30-60 min
- Smoke tests: 30-60 min

### Per Training Run (~1-14 days)
Depends on dataset size and epochs:
- Small dataset (1K samples): 1-2 days
- Medium dataset (10K samples): 3-7 days
- Large dataset (100K samples): 1-2 weeks

### Evaluation (~1-4 hours)
- 100 samples: ~10 minutes
- 1,000 samples: ~1 hour
- 10,000 samples: ~4-8 hours

### Total First Deployment
**Minimum:** ~1 week (setup + small training + eval)
**Typical:** ~2 weeks (setup + medium training + eval + iteration)

---

## Cost Optimization Tips

1. **Front-load CPU Testing:** Catch 80% of issues before GPU deployment
2. **Use Smoke Tests:** Test with minimal data first (10 steps, 10 samples)
3. **Monitor Actively:** Don't let failed runs continue for hours
4. **Checkpoint Frequently:** Save every N steps to avoid losing progress
5. **Validate Data First:** Ensure NeMo JSONL correct before training
6. **Test Incrementally:** 10 steps → 100 steps → 1000 steps → full

---

## Checklist: Ready for GPU Testing?

Before scheduling GPU server time, ensure:

- [ ] ✅ All CPU tests passing
- [ ] ✅ Data pipeline validated end-to-end on CPU
- [ ] ✅ NeMo JSONL format verified (without tokenizer)
- [ ] ✅ Configuration files validated
- [ ] ✅ Environment variables set correctly
- [ ] ✅ Scripts have no import errors
- [ ] ✅ Logging and manifest systems working
- [ ] ✅ Git repository clean and committed
- [ ] ✅ Documentation reviewed
- [ ] ✅ GPU deployment checklist reviewed

**If all checked:** Proceed to GPU server deployment

**If any unchecked:** Fix on CPU server first to minimize GPU costs

---

## Next Steps

1. **Complete CPU testing:** Run `scripts/cpu_comprehensive_test.sh`
2. **Review results:** Check `CPU_TEST_RESULTS.md`
3. **Fix any failures:** Address issues found in CPU testing
4. **Prepare data:** Finalize datasets for GPU server
5. **Read GPU guide:** Review `GPU_DEPLOYMENT_CHECKLIST.md`
6. **Schedule GPU time:** Estimate based on timeline above
7. **Deploy to GPU:** Follow step-by-step checklist

---

## Questions to Answer Before GPU Deployment

1. **How large is the training dataset?** ___________ samples
2. **How many epochs planned?** ___________ epochs
3. **What is acceptable training time?** ___________ days
4. **Is model download complete?** YES / NO
5. **Are data files ready to transfer?** YES / NO
6. **Is API key configured?** YES / NO
7. **Is monitoring setup (WandB)?** YES / NO / NOT NEEDED
8. **Who will monitor training?** ___________
9. **What is fallback plan if training fails?** ___________
10. **Are results expectations documented?** YES / NO

---

**Remember:** Thorough CPU testing now saves expensive GPU server time later!
