# Quick Installation Guide

## TL;DR - Fresh Server Setup

```bash
# 1. Clone
cd /opt
git clone <your-repo> NeMo-Charlie
cd NeMo-Charlie

# 2. Configure
cp .env.example .env
nano .env  # Set CPU_ONLY_MODE=true or false

# 3. One-command setup (10-15 minutes)
bash scripts/setup_env_v2.sh

# 4. Start working
source venv/bin/activate
```

**Done!** Setup includes installation, patching, and validation.

## What You Get

### CPU Mode (Development)
```bash
# In .env
CPU_ONLY_MODE=true
```
- PyTorch (CPU)
- zarr 2.x
- NeMo 2.5.2 (patched)
- Only "NeMo-Run" warning (expected)
- Time: ~5-10 minutes

### GPU Mode (Production)
```bash
# In .env
CPU_ONLY_MODE=false
```
- PyTorch 2.9.0+cu128
- zarr 2.x
- transformer-engine
- NeMo 2.5.2 (patched)
- Only "NeMo-Run" warning (expected)
- Time: ~10-15 minutes

## Setup Features

✅ **Fully Automated** - No prompts, no interaction  
✅ **Optimal Install Order** - Minimizes warnings  
✅ **Integrated Validation** - Runs automatically at end  
✅ **Configuration-Driven** - CPU vs GPU via .env  
✅ **Clean Slate Required** - Fails if venv/ exists  

## Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git build-essential

# For GPU mode
nvidia-smi  # Verify CUDA 12.x
```

## Troubleshooting

**"venv/ already exists"**
```bash
rm -rf venv/ && bash scripts/setup_env_v2.sh
```

**"CUDA not found" (GPU mode)**
```bash
# Switch to CPU mode
nano .env  # Set CPU_ONLY_MODE=true
bash scripts/setup_env_v2.sh
```

## For Full Documentation

See `README.md` for:
- Model conversion
- Data preparation
- Training
- Evaluation
- Configuration details
- Advanced topics

## Files Reference

- `scripts/setup_env_v2.sh` - Main setup script
- `scripts/validate_environment.py` - Validation (auto-run)
- `scripts/apply_nemo_patches.py` - Patches (auto-run)
- `requirements.txt` - Base dependencies
- `requirements_nemo.txt` - NeMo and supporting packages
- `requirements_optional.txt` - Optional performance packages
- `NEMO_FIXES.md` - Detailed patch documentation
