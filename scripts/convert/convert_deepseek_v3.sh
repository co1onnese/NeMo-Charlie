#!/usr/bin/env bash

set -euo pipefail

# Converts DeepSeek-V3 FP8 checkpoint downloaded from Hugging Face into a BF16
# checkpoint suitable for NeMo import using optimized parallel conversion.
#
# This script automatically selects the best conversion mode:
# - Multi-GPU mode (6-8x faster) if multiple GPUs available and torchrun detected
# - Single-GPU optimized (2-3x faster) otherwise
#
# Performance:
#   Legacy (sequential):   ~25-30 minutes (deprecated)
#   Single-GPU optimized:  ~10-15 minutes (default)
#   Multi-GPU optimized:   ~4-6 minutes (with 8 GPUs)

usage() {
  cat <<'USAGE'
Usage:
  scripts/convert/convert_deepseek_v3.sh \
    --source /path/to/deepseek-ai/DeepSeek-V3-Base \
    --output /path/to/output/bf16

Options:
  --source       Path to cloned FP8 model repository (HuggingFace format)
  --output       Destination directory for BF16 safetensors
  --log          Optional log file path (default: logs/conversion/deepseek_v3.log)
  --mode         Conversion mode: auto|single|multi|legacy (default: auto)
  --max-workers  I/O worker threads per GPU (default: 8 for single, 4 for multi)

Modes:
  auto    - Automatically select best mode based on available GPUs
  single  - Optimized single-GPU with async I/O (2-3x speedup)
  multi   - Multi-GPU distributed conversion (6-8x speedup, requires torchrun)
  legacy  - Original sequential script (deprecated, for comparison only)

Examples:
  # Automatic mode selection (recommended)
  ./convert_deepseek_v3.sh --source /data/fp8 --output /data/bf16

  # Force multi-GPU mode with torchrun
  CONVERSION_MODE=multi ./convert_deepseek_v3.sh --source /data/fp8 --output /data/bf16

  # Use legacy script for comparison
  ./convert_deepseek_v3.sh --source /data/fp8 --output /data/bf16 --mode legacy
USAGE
}

SOURCE=""
OUTPUT=""
LOG_FILE="logs/conversion/deepseek_v3.log"
MODE="${CONVERSION_MODE:-auto}"
MAX_WORKERS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SOURCE" || -z "$OUTPUT" ]]; then
  echo "Error: --source and --output are required" >&2
  usage >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

echo "[INFO] ========================================" | tee "$LOG_FILE"
echo "[INFO] DeepSeek-V3 FP8→BF16 Conversion" | tee -a "$LOG_FILE"
echo "[INFO] ========================================" | tee -a "$LOG_FILE"
echo "[INFO] Source: $SOURCE" | tee -a "$LOG_FILE"
echo "[INFO] Output: $OUTPUT" | tee -a "$LOG_FILE"
echo "[INFO]" | tee -a "$LOG_FILE"

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
  GPU_COUNT=0
fi

# Auto-select conversion mode
if [[ "$MODE" == "auto" ]]; then
  if [[ $GPU_COUNT -ge 4 ]] && command -v torchrun &> /dev/null; then
    MODE="multi"
    echo "[INFO] Auto-selected: Multi-GPU mode ($GPU_COUNT GPUs detected)" | tee -a "$LOG_FILE"
  elif [[ $GPU_COUNT -ge 1 ]]; then
    MODE="single"
    echo "[INFO] Auto-selected: Single-GPU optimized mode" | tee -a "$LOG_FILE"
  else
    echo "[WARNING] No GPUs detected, falling back to legacy mode" | tee -a "$LOG_FILE"
    MODE="legacy"
  fi
fi

# Set default max_workers based on mode
if [[ -z "$MAX_WORKERS" ]]; then
  case "$MODE" in
    single)
      MAX_WORKERS=8
      ;;
    multi)
      MAX_WORKERS=4
      ;;
    *)
      MAX_WORKERS=1
      ;;
  esac
fi

# Execute conversion based on selected mode
case "$MODE" in
  multi)
    echo "[INFO] Running MULTI-GPU conversion (optimized)" | tee -a "$LOG_FILE"
    echo "[INFO] Using $GPU_COUNT GPUs with $MAX_WORKERS I/O workers each" | tee -a "$LOG_FILE"
    echo "[INFO] Expected time: ~4-6 minutes" | tee -a "$LOG_FILE"
    echo "[INFO]" | tee -a "$LOG_FILE"
    
    torchrun --nproc_per_node=$GPU_COUNT scripts/convert/fp8_cast_bf16_parallel.py \
      --input-fp8-hf-path "$SOURCE" \
      --output-bf16-hf-path "$OUTPUT" \
      --multi-gpu \
      --max-workers $MAX_WORKERS \
      2>&1 | tee -a "$LOG_FILE"
    ;;
    
  single)
    echo "[INFO] Running SINGLE-GPU conversion (optimized)" | tee -a "$LOG_FILE"
    echo "[INFO] Using async I/O with $MAX_WORKERS workers" | tee -a "$LOG_FILE"
    echo "[INFO] Expected time: ~10-15 minutes" | tee -a "$LOG_FILE"
    echo "[INFO]" | tee -a "$LOG_FILE"
    
    python3 scripts/convert/fp8_cast_bf16_parallel.py \
      --input-fp8-hf-path "$SOURCE" \
      --output-bf16-hf-path "$OUTPUT" \
      --max-workers $MAX_WORKERS \
      2>&1 | tee -a "$LOG_FILE"
    ;;
    
  legacy)
    echo "[WARNING] Running LEGACY sequential conversion (DEPRECATED)" | tee -a "$LOG_FILE"
    echo "[WARNING] This mode is 2-8x SLOWER than optimized modes" | tee -a "$LOG_FILE"
    echo "[WARNING] Expected time: ~25-30 minutes" | tee -a "$LOG_FILE"
    echo "[WARNING] Use --mode single or --mode multi for better performance" | tee -a "$LOG_FILE"
    echo "[INFO]" | tee -a "$LOG_FILE"
    
    python3 scripts/convert/fp8_cast_bf16.py \
      --input-fp8-hf-path "$SOURCE" \
      --output-bf16-hf-path "$OUTPUT" \
      2>&1 | tee -a "$LOG_FILE"
    ;;
    
  *)
    echo "[ERROR] Invalid mode: $MODE" | tee -a "$LOG_FILE"
    echo "[ERROR] Valid modes: auto, single, multi, legacy" | tee -a "$LOG_FILE"
    exit 1
    ;;
esac

echo "[INFO] Weight conversion complete. Copying configuration files..." | tee -a "$LOG_FILE"

# Copy required configuration and tokenizer files
# Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html
REQUIRED_FILES=(
  "tokenizer_config.json"
  "tokenizer.json"
  "modeling_deepseek.py"
  "configuration_deepseek.py"
)

for file in "${REQUIRED_FILES[@]}"; do
  if [[ -f "$SOURCE/$file" ]]; then
    echo "[INFO] Copying $file..." | tee -a "$LOG_FILE"
    cp "$SOURCE/$file" "$OUTPUT/"
  else
    echo "[WARNING] File not found: $SOURCE/$file (skipping)" | tee -a "$LOG_FILE"
  fi
done

# Copy and modify config.json to remove quantization_config
# This is required because NeMo doesn't support FP8 quantization format
if [[ -f "$SOURCE/config.json" ]]; then
  echo "[INFO] Creating config.json (removing quantization_config)..." | tee -a "$LOG_FILE"

  # Check if jq is available
  if command -v jq &> /dev/null; then
    jq 'del(.quantization_config)' "$SOURCE/config.json" > "$OUTPUT/config.json"
  else
    # Fallback: use python to remove quantization_config
    python3 -c "
import json
import sys
with open('$SOURCE/config.json', 'r') as f:
    config = json.load(f)
config.pop('quantization_config', None)
with open('$OUTPUT/config.json', 'w') as f:
    json.dump(config, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"
  fi
  echo "[INFO] config.json created" | tee -a "$LOG_FILE"
else
  echo "[ERROR] config.json not found in source directory" | tee -a "$LOG_FILE"
  exit 1
fi

# Verify all required files exist in output
echo "[INFO] Verifying output directory..." | tee -a "$LOG_FILE"
VERIFICATION_FILES=("config.json" "tokenizer.json")
MISSING_FILES=()

for file in "${VERIFICATION_FILES[@]}"; do
  if [[ ! -f "$OUTPUT/$file" ]]; then
    MISSING_FILES+=("$file")
  fi
done

if [[ ${#MISSING_FILES[@]} -gt 0 ]]; then
  echo "[ERROR] Missing required files in output: ${MISSING_FILES[*]}" | tee -a "$LOG_FILE"
  exit 1
fi

cat <<EOF | tee -a "$LOG_FILE"
[INFO] Conversion complete.
[INFO] BF16 checkpoint stored at: $OUTPUT
[INFO] All required configuration files copied.
EOF

