# NeMo-Charlie Setup Guide

Complete setup instructions for a fresh machine installation.

## Prerequisites

- **Python 3.8+** (3.10 or 3.12 recommended)
- **Git** for cloning the repository
- **CUDA 12.4** (only if using GPU)
- **16GB+ RAM** for basic operations
- **100GB+ disk space** for models and data

## Quick Start (Recommended)

For a **fresh machine** with NeMo support:

```bash
# Clone the repository
git clone https://github.com/co1onnese/NeMo-Charlie.git
cd NeMo-Charlie

# Run the comprehensive setup script
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

This single command:
1. ✅ Creates a Python virtual environment
2. ✅ Installs PyTorch with GPU support
3. ✅ Installs all dependencies
4. ✅ Installs NeMo Framework
5. ✅ Automatically applies NeMo patches
6. ✅ Verifies everything works

**Time:** ~10-15 minutes on a good connection

## Setup Options

### Option 1: Full Installation (GPU + NeMo)

For training and model conversion:

```bash
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

**What you get:**
- PyTorch with CUDA 12.4 support
- NeMo Framework with patches
- All training and evaluation tools
- DeepSeek-V3 model support

### Option 2: CPU-Only with NeMo

For testing without GPU:

```bash
INSTALL_NEMO=true bash scripts/setup_env.sh
```

**What you get:**
- PyTorch CPU version
- NeMo Framework with patches
- Can test conversion scripts
- Cannot train large models

### Option 3: Base Installation Only

For data processing without NeMo:

```bash
bash scripts/setup_env.sh
```

**What you get:**
- PyTorch CPU version
- Data processing tools
- Evaluation and backtesting
- No model training capabilities

## What Happens During Setup

### Step 1: Virtual Environment

Creates an isolated Python environment in `venv/`:
```
Creating virtual environment in venv...
✓ Created virtual environment
✓ Virtual environment activated
```

### Step 2: PyTorch Installation

Installs PyTorch based on your selection:
- **GPU:** CUDA 12.4 optimized build (~2GB download)
- **CPU:** Lightweight build (~200MB download)

### Step 3: Base Dependencies

Installs core packages from `requirements.txt`:
- transformers (HuggingFace)
- datasets
- pandas, numpy
- python-dotenv
- tqdm, etc.

### Step 4: NeMo Framework

Installs NeMo and dependencies from `requirements_nemo.txt`:
- nemo-toolkit
- megatron-core
- lightning, pytorch-lightning
- omegaconf, hydra-core
- Optional packages (may warn if fail)

**Expected warnings:**
```
[WARNING] Some NeMo dependencies failed (exit code: 1)
[WARNING] This is often due to optional packages like:
           - mamba-ssm (requires C++ compilation)
           - flash-attn (requires CUDA)
           - transformer-engine (optional optimization)
```

These warnings are **normal** and **expected**. The core NeMo functionality will still work.

### Step 5: NeMo Patches (Automatic!)

This is the **key improvement** - patches are now applied automatically:

```
========================================
Step 5: Applying NeMo Patches
========================================

NeMo 2.5.2 has import issues with optional dependencies:
  - nv_one_logger (telemetry - not on public PyPI)
  - nemo_run (recipes - not on public PyPI)
  - tensorstore (export - optional)

Applying patches to make these imports conditional...

[1/6] Patching one_logger_callback.py...
  ✓ Patched successfully
...
✓ NeMo patches applied successfully
```

**What this fixes:**
- `ModuleNotFoundError: No module named 'nv_one_logger'`
- `ModuleNotFoundError: No module named 'nemo_run'`
- `ModuleNotFoundError: No module named 'tensorstore'`

### Step 6: Verification

Comprehensive testing of all installations:

```
Core dependencies:
  ✓ PyTorch: 2.x.x
  ✓ Transformers: 4.x.x
  ✓ Datasets: 2.x.x

NeMo Framework:
  ✓ NeMo: 2.5.2
  ✓ Megatron-Core: 0.x.x

NeMo LLM Module (Critical Test):
  ✓ nemo.collections.llm: importable
  ✓ llm.import_ckpt: available
  ✓ llm.DeepSeekV3Config: available
  ✓ llm.DeepSeekModel: available

✓ All verifications passed
```

## After Setup

### Activate the Environment

Every time you start a new terminal session:

```bash
cd /path/to/NeMo-Charlie
source venv/bin/activate
```

You'll see `(venv)` in your prompt.

### Verify NeMo Still Works

If you update NeMo or reinstall dependencies:

```bash
source venv/bin/activate
bash scripts/verify_nemo_fixes.sh
```

### Test Model Conversion

```bash
source venv/bin/activate
python scripts/convert/import_to_nemo.py --help
```

Should show the help message without errors.

## Troubleshooting

### Problem: "No module named 'nv_one_logger'"

**Cause:** NeMo patches weren't applied or were overwritten

**Solution:**
```bash
source venv/bin/activate
python scripts/apply_nemo_patches.py
```

### Problem: "CUDA out of memory"

**Cause:** Model too large for your GPU

**Solutions:**
1. Use tensor parallelism (requires multiple GPUs)
2. Use a smaller model
3. Reduce batch size

### Problem: "pip install failed"

**Cause:** Network issues or missing system dependencies

**Solution:**
```bash
# Try again with verbose output
pip install -r requirements_nemo.txt -v

# Or install packages individually
pip install nemo-toolkit
pip install megatron-core
# etc.
```

### Problem: Setup script hangs

**Cause:** Large downloads (PyTorch ~2GB, NeMo dependencies ~1GB)

**Solution:**
- Be patient, especially on slower connections
- Monitor with `htop` or `nvidia-smi` (if GPU)
- Check disk space: `df -h`

### Problem: "Permission denied"

**Cause:** Trying to install system-wide or in read-only directory

**Solution:**
```bash
# Ensure you're in the project directory
cd /path/to/NeMo-Charlie

# Don't use sudo
bash scripts/setup_env.sh
```

## Starting Fresh

If setup fails or environment is corrupted:

```bash
# Remove the virtual environment
rm -rf venv/

# Re-run setup
INSTALL_NEMO=true INSTALL_GPU_TORCH=true bash scripts/setup_env.sh
```

## Advanced: Manual Setup

If the automated script doesn't work, see `NEMO_FIXES.md` for:
- Manual installation steps
- Manual patch application
- Detailed troubleshooting

## Next Steps

After successful setup:

1. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

2. **Test model conversion:**
   ```bash
   python scripts/convert/import_to_nemo.py --help
   ```

3. **Run data processing:**
   ```bash
   python src/parsers/xml_to_jsonl.py --help
   ```

4. **Run full pipeline:**
   ```bash
   bash scripts/run_full_pipeline.sh
   ```

## Support

- **Issues with NeMo patches:** See `NEMO_FIXES.md`
- **General setup problems:** Check this guide's Troubleshooting section
- **Pipeline issues:** See `README.md` and `runbook/README.md`

## Summary of Files

- **`scripts/setup_env.sh`** - Main setup script (run this!)
- **`scripts/apply_nemo_patches.py`** - NeMo patching (automatic)
- **`scripts/verify_nemo_fixes.sh`** - Verify patches work
- **`NEMO_FIXES.md`** - Detailed patch documentation
- **`SETUP_GUIDE.md`** - This file
- **`README.md`** - Project overview
