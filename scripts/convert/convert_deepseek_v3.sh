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

cat <<EOF | tee -a "$LOG_FILE"
[INFO] Conversion complete.
[INFO] BF16 checkpoint stored at: $OUTPUT
EOF

