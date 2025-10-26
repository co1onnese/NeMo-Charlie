# Model Storage Guide - DeepSeek-V3.2-Exp

## 📊 Current Status

### Model Download Status: ❌ **NOT DOWNLOADED YET**

```
HuggingFace Cache: /root/.cache/huggingface
Status: Directory does not exist
DeepSeek Model: Not downloaded
```

### Disk Space Available: **38 GB** (of 48 GB total)

⚠️ **WARNING**: Current disk space (38 GB) may be insufficient for full model download (~20-40 GB)

---

## 📦 Storage Locations

### 1. Base Model Weights (Downloaded from HuggingFace)

**Location:**
```
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3.2-Exp/
```

**Structure:**
```
/root/.cache/huggingface/
├── hub/
│   └── models--deepseek-ai--DeepSeek-V3.2-Exp/
│       ├── snapshots/
│       │   └── [commit-hash]/
│       │       ├── config.json
│       │       ├── tokenizer.json
│       │       ├── tokenizer_config.json
│       │       ├── model-00001-of-00008.safetensors
│       │       ├── model-00002-of-00008.safetensors
│       │       └── ... (8 total shard files)
│       └── refs/
└── ...
```

**Size:** ~20-40 GB (depends on model precision)
- FP16: ~20 GB
- BF16: ~20 GB  
- FP32: ~40 GB

**When Downloaded:**
- Automatically on first use when running:
  - `tokenize_and_shard.py`
  - `train_sft.py`
  - `evaluate_sft.py`

**Control Location:**
You can change the cache location by setting environment variables:

```bash
# Option 1: Set in .env
HF_HOME=/path/to/large/disk/huggingface

# Option 2: Set before running
export HF_HOME=/path/to/large/disk/huggingface
python3 src/train/train_sft.py --config configs/sft_config.yaml
```

---

### 2. Fine-Tuned Model Weights (LoRA Adapters)

**Location:**
```
/opt/SFT-Charlie/checkpoints/sft-deepseek-v3.2exp-longctx/
```

**Structure:**
```
checkpoints/sft-deepseek-v3.2exp-longctx/
├── adapter_config.json          # LoRA configuration
├── adapter_model.safetensors    # ✅ Main fine-tuned weights (~100-500 MB)
├── config.json                  # Model config (copied from base)
├── tokenizer.json               # Extended tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── training_args.bin            # Training hyperparameters
├── manifest.json                # Reproducibility info
├── trainer_state.json           # Training state
├── checkpoint-500/              # Intermediate checkpoint
│   ├── adapter_model.safetensors
│   └── ...
├── checkpoint-1000/
└── checkpoint-1500/
```

**Size:** ~100-500 MB for LoRA adapters
- Much smaller than full model (20 GB+) because we only train adapters
- Checkpoints: ~100-500 MB each (x3 max = ~1.5 GB total)

**Configured in:** `.env` file
```bash
OUTPUT_DIR=checkpoints/sft-deepseek-v3.2exp-longctx
```

**Contains:**
- LoRA adapter weights (the fine-tuned parameters)
- Extended tokenizer with special tokens
- Training configuration
- Reproducibility manifest

**⚠️ Note:** The base model weights are NOT duplicated here. Only the small LoRA adapters are saved.

---

## 💾 Complete Storage Breakdown

### On CPU Server (Current):
```
Location                                    Size        Status
───────────────────────────────────────────────────────────────
/opt/SFT-Charlie/                          ~50 MB      ✅ Exists
  ├── src/ (code)                          ~2 MB       ✅
  ├── data/samples/                        ~10 KB      ✅
  ├── data/jsonl/smoke_test.jsonl          ~8 KB       ✅
  └── logs/                                ~100 KB     ✅

Total Used: ~50 MB
Available: 38 GB
```

### After Full Pipeline (GPU Server):
```
Location                                    Size        Notes
───────────────────────────────────────────────────────────────────
/root/.cache/huggingface/                  20-40 GB    Base model
/opt/SFT-Charlie/data/
  ├── raw_xml/                             ~10 MB      Your 20 XML files
  ├── jsonl/                               ~15 MB      Converted data
  ├── hf_datasets/sft_dataset/             ~20 MB      Arrow format
  ├── hf_datasets/sft_dataset_tokenized/   2-5 GB      Tokenized
  └── price_cache/                         ~50 MB      Cached prices

/opt/SFT-Charlie/checkpoints/
  └── sft-deepseek-v3.2exp-longctx/        ~2 GB       Fine-tuned model
      ├── adapter_model.safetensors        ~300 MB     LoRA weights
      └── checkpoint-*/                    ~1.5 GB     3 checkpoints

/opt/SFT-Charlie/results/                  ~5 MB       Evaluation
/opt/SFT-Charlie/logs/                     ~100 MB     Training logs

───────────────────────────────────────────────────────────────────
TOTAL REQUIRED: 25-50 GB
```

---

## 🚀 Model Download Process

### When Does It Download?

The model is **lazily downloaded** when first accessed:

1. **During Tokenization:**
   ```bash
   python3 src/data/tokenize_and_shard.py
   ```
   Downloads: Tokenizer files only (~10 MB)

2. **During Training:**
   ```bash
   python3 src/train/train_sft.py --config configs/sft_config.yaml
   ```
   Downloads: Full model weights (~20-40 GB)

3. **During Evaluation:**
   ```bash
   python3 src/eval/evaluate_sft.py --model_dir [path]
   ```
   Uses: Already downloaded model OR fine-tuned adapters

### Download Progress

HuggingFace shows progress bars:
```
Downloading model.safetensors: 100%|████████| 20.1G/20.1G [10:32<00:00, 32.0MB/s]
```

### Speed Depends On:
- Internet connection (~30 MB/s typical)
- HuggingFace CDN availability
- Time: ~10-30 minutes for full model

---

## ⚙️ Configuration Options

### 1. Change HuggingFace Cache Location

**Add to .env:**
```bash
# Use a disk with more space
HF_HOME=/mnt/large_disk/huggingface_cache
```

**Or set before running:**
```bash
export HF_HOME=/mnt/large_disk/huggingface_cache
python3 src/train/train_sft.py --config configs/sft_config.yaml
```

### 2. Change Output Directory

**In .env:**
```bash
OUTPUT_DIR=/mnt/large_disk/checkpoints/my-model
```

### 3. Reduce Checkpoint Storage

**Edit configs/sft_config.yaml:**
```yaml
save_total_limit: 1  # Keep only 1 checkpoint (instead of 3)
save_steps: 5000     # Save less frequently
```

---

## 💡 Storage Optimization Tips

### For Limited Disk Space:

1. **Use External Storage:**
   ```bash
   # Mount external drive
   sudo mount /dev/sdb1 /mnt/external
   
   # Set cache location
   export HF_HOME=/mnt/external/huggingface
   ```

2. **Reduce Checkpoints:**
   ```yaml
   # In sft_config.yaml
   save_total_limit: 1  # Only keep latest checkpoint
   ```

3. **Don't Keep Tokenized Dataset:**
   ```bash
   # Delete after training starts
   rm -rf data/hf_datasets/sft_dataset_tokenized/
   ```

4. **Clean Up After Training:**
   ```bash
   # Remove intermediate checkpoints (keep only final)
   rm -rf checkpoints/*/checkpoint-*/
   ```

5. **Use 4-bit Quantization:**
   ```yaml
   # Already configured in sft_config.yaml
   load_in_4bit: true  # Reduces memory (not disk) usage
   ```

---

## 📁 What Gets Saved Where

### Base Model (From HuggingFace):
- **Where:** `/root/.cache/huggingface/`
- **Size:** 20-40 GB
- **Contains:** Original DeepSeek weights
- **Shareable:** Yes (same for all projects)
- **Downloaded:** Once, reused by all projects

### Fine-Tuned Model (Your Training):
- **Where:** `/opt/SFT-Charlie/checkpoints/[model-name]/`
- **Size:** 100-500 MB (LoRA adapters only!)
- **Contains:** LoRA weights + extended tokenizer
- **Shareable:** Yes (portable to other systems)
- **Usage:** Loaded on top of base model

### To Use Fine-Tuned Model:
You need **BOTH**:
1. Base model (20 GB in cache)
2. LoRA adapters (500 MB in checkpoints)

Total: ~20.5 GB, but base model is shared across projects.

---

## 🔍 Checking Storage

### Check HuggingFace Cache:
```bash
ls -lh ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/
```

### Check Model Downloaded:
```bash
ls -lh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3.2-Exp/
```

### Check Fine-Tuned Model:
```bash
ls -lh /opt/SFT-Charlie/checkpoints/sft-deepseek-v3.2exp-longctx/
```

### Check Disk Space:
```bash
df -h /
df -h ~/.cache/
```

---

## 🎯 Recommendations

### For Current CPU Server (38 GB available):

⚠️ **DO NOT download model here** - insufficient space

**Action:**
1. ✅ Keep current setup (code + small data)
2. ✅ Process XML files here (small)
3. ❌ Don't run tokenization (requires model download)
4. ❌ Don't run training

### For GPU Server:

**Minimum Requirements:**
- **50 GB** free disk space
- Or 100 GB to be safe (for logs, checkpoints, etc.)

**Recommended Setup:**
```bash
# Check space
df -h /

# If limited, set custom cache location
export HF_HOME=/mnt/large_disk/hf_cache

# Or add to .env:
echo "HF_HOME=/mnt/large_disk/hf_cache" >> .env
```

---

## 🔄 Model Download Commands

### Manual Download (Optional):

If you want to pre-download the model:

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download model
huggingface-cli download deepseek-ai/DeepSeek-V3.2-Exp

# Check download
ls -lh ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3.2-Exp/
```

### Verify Before Training:

```python
python3 << 'EOF'
from transformers import AutoTokenizer

# This will download tokenizer if not cached
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-V3.2-Exp",
    trust_remote_code=True
)

print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
EOF
```

---

## 📌 Summary

| Question | Answer |
|----------|--------|
| **Model downloaded yet?** | ❌ NO |
| **Where will it download?** | `/root/.cache/huggingface/hub/` |
| **How big?** | ~20-40 GB |
| **Where are fine-tuned weights?** | `/opt/SFT-Charlie/checkpoints/sft-deepseek-v3.2exp-longctx/` |
| **How big are LoRA adapters?** | ~100-500 MB |
| **Current disk space?** | 38 GB (⚠️ too small for model) |
| **Recommended?** | Use GPU server with 50+ GB free |

---

## ✅ Action Items

1. **Now (CPU Server):**
   - ✅ Code is ready
   - ✅ Process XML files (small)
   - ❌ Don't download model

2. **On GPU Server:**
   - ✅ Check disk space (need 50+ GB)
   - ✅ Set HF_HOME if needed
   - ✅ Run pipeline (model auto-downloads)

3. **After Training:**
   - ✅ Fine-tuned model in `checkpoints/`
   - ✅ Size: ~500 MB
   - ✅ Portable (copy to other systems)
   - ✅ Needs base model to run

---

**Last Updated:** October 26, 2025
