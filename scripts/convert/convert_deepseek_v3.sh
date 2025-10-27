#!/usr/bin/env bash

set -euo pipefail

# Converts DeepSeek-V3 FP8 checkpoint downloaded from Hugging Face into a BF16
# checkpoint suitable for NeMo import. Also prepares metadata describing the
# conversion.

usage() {
  cat <<'USAGE'
Usage:
  scripts/convert/convert_deepseek_v3.sh \
    --source /path/to/deepseek-ai/DeepSeek-V3-Base \
    --output /path/to/output/bf16

Options:
  --source   Path to cloned FP8 model repository (HuggingFace format)
  --output   Destination directory for BF16 safetensors
  --log      Optional log file path (default: logs/conversion/deepseek_v3.log)
USAGE
}

SOURCE=""
OUTPUT=""
LOG_FILE="logs/conversion/deepseek_v3.log"

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

echo "[INFO] Starting DeepSeek FP8→BF16 conversion" | tee "$LOG_FILE"
echo "[INFO] Source: $SOURCE" | tee -a "$LOG_FILE"
echo "[INFO] Output: $OUTPUT" | tee -a "$LOG_FILE"

python3 scripts/convert/fp8_cast_bf16.py \
  --input-fp8-hf-path "$SOURCE" \
  --output-bf16-hf-path "$OUTPUT" \
  2>&1 | tee -a "$LOG_FILE"

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

