#!/bin/bash
# run_full_pipeline.sh
# Complete end-to-end pipeline for SFT training
#
# Pipeline Stages:
#   0. Model Setup (Download & Convert DeepSeek-V3)
#   1. Data Preparation (XML → JSONL → HF Dataset → NeMo Dataset)
#   2. Training (NeMo fine-tuning)
#   3. Evaluation (Model predictions + backtesting)

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

echo "========================================="
echo "SFT Trading Pipeline - Full Run"
echo "========================================="
echo "Start time: $(date)"
echo ""

# Load environment variables
if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "Warning: .env file not found. Using defaults."
fi

# Activate venv
if [ -d "venv/bin" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Warning: venv/bin not found. Make sure dependencies are installed."
fi

echo ""

# Parse arguments
SMOKE_TEST=false
SKIP_MODEL_SETUP=false
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --skip-model-setup)
            SKIP_MODEL_SETUP=true
            shift
            ;;
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--smoke-test] [--skip-model-setup] [--skip-data] [--skip-train] [--skip-eval]"
            exit 1
            ;;
    esac
done

if [ "$SMOKE_TEST" = true ]; then
    echo "SMOKE TEST MODE"
    export CPU_ONLY_MODE=true
    export SMOKE_TEST_MODE=true
fi

# Step 0: Model Setup (Download & Convert DeepSeek-V3)
if [ "$SKIP_MODEL_SETUP" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 0: Model Setup"
    echo "========================================="

    # Define model paths (using /data for large files)
    MODEL_SOURCE_DIR=${MODEL_SOURCE_DIR:-/data/models/deepseek-v3-source}
    MODEL_BF16_DIR=${MODEL_BF16_DIR:-/data/models/deepseek-v3-bf16}
    MODEL_NEMO_PATH=${MODEL_NEMO_PATH:-/data/models/deepseek-v3-base_tp8_pp1.nemo}

    # Check if NeMo model already exists
    if [ -f "$MODEL_NEMO_PATH" ]; then
        echo "✓ NeMo model already exists at: $MODEL_NEMO_PATH"
        echo "  Skipping model download and conversion."
        echo "  To re-convert, delete the file and re-run without --skip-model-setup"
    else
        echo "NeMo model not found. Starting download and conversion process..."
        echo ""

        # 0.1: Download DeepSeek-V3 from HuggingFace
        if [ ! -d "$MODEL_SOURCE_DIR" ] || [ ! -f "$MODEL_SOURCE_DIR/config.json" ]; then
            echo "0.1: Downloading DeepSeek-V3-Base from HuggingFace..."
            echo "     This will download ~100GB of model weights."
            echo ""

            # Check if git-lfs is installed
            if ! command -v git-lfs &> /dev/null; then
                echo "Error: git-lfs is not installed."
                echo "Please install git-lfs first:"
                echo "  Ubuntu/Debian: sudo apt-get install git-lfs"
                echo "  CentOS/RHEL:   sudo yum install git-lfs"
                echo "  macOS:         brew install git-lfs"
                exit 1
            fi

            # Initialize git-lfs
            git lfs install

            # Create parent directory
            mkdir -p "$(dirname "$MODEL_SOURCE_DIR")"

            # Clone with LFS
            echo "Cloning deepseek-ai/DeepSeek-V3-Base..."
            echo "NOTE: This may take 30-60 minutes depending on your connection."
            git clone https://huggingface.co/deepseek-ai/DeepSeek-V3-Base "$MODEL_SOURCE_DIR"

            echo "✓ Download complete"
        else
            echo "0.1: Download already complete"
            echo "     ✓ DeepSeek-V3 source exists at: $MODEL_SOURCE_DIR"
            echo "     To re-download: rm -rf $MODEL_SOURCE_DIR"
        fi

        echo ""

        # 0.2: Convert FP8 to BF16 (with optimized parallel conversion)
        # Check if BF16 conversion is already complete by verifying required files
        BF16_COMPLETE=true
        REQUIRED_BF16_FILES=("config.json" "tokenizer.json")

        if [ ! -d "$MODEL_BF16_DIR" ]; then
            BF16_COMPLETE=false
        else
            for file in "${REQUIRED_BF16_FILES[@]}"; do
                if [ ! -f "$MODEL_BF16_DIR/$file" ]; then
                    echo "   Missing required file: $file"
                    BF16_COMPLETE=false
                    break
                fi
            done
        fi

        if [ "$BF16_COMPLETE" = false ]; then
            echo "0.2: Converting FP8 weights to BF16 (optimized parallel conversion)..."
            
            # Detect available GPUs
            if command -v nvidia-smi &> /dev/null; then
                GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
            else
                GPU_COUNT=0
            fi
            
            # Display expected time based on mode
            if [ $GPU_COUNT -ge 4 ] && command -v torchrun &> /dev/null; then
                echo "     Mode: Multi-GPU (${GPU_COUNT} GPUs detected)"
                echo "     Expected time: ~4-6 minutes (6-8x faster)"
            elif [ $GPU_COUNT -ge 1 ]; then
                echo "     Mode: Single-GPU optimized"
                echo "     Expected time: ~10-15 minutes (2-3x faster)"
            else
                echo "     Mode: Legacy sequential (no GPU detected)"
                echo "     Expected time: ~25-30 minutes"
            fi
            echo ""

            # Check if we're in CPU-only mode
            if [ "${CPU_ONLY_MODE:-false}" = "true" ]; then
                echo "Warning: CPU_ONLY_MODE is enabled, but conversion requires GPU."
                echo "Skipping conversion. Please run without --smoke-test on a GPU server."
            else
                # Use optimized conversion (automatically selects best mode)
                # Set CONVERSION_MODE env var to override: single|multi|legacy
                bash scripts/convert/convert_deepseek_v3.sh \
                    --source "$MODEL_SOURCE_DIR" \
                    --output "$MODEL_BF16_DIR" \
                    --log logs/conversion/deepseek_v3_fp8_to_bf16.log

                echo "✓ BF16 conversion complete"
            fi
        else
            echo "0.2: BF16 conversion already complete"
            echo "     ✓ BF16 model exists at: $MODEL_BF16_DIR"
            echo "     ✓ All required files present (config.json, tokenizer.json, etc.)"
            echo "     To re-convert: rm -rf $MODEL_BF16_DIR"
        fi

        echo ""

        # 0.3: Import to NeMo format
        if [ ! -f "$MODEL_NEMO_PATH" ]; then
            echo "0.3: Importing BF16 checkpoint to NeMo format..."
            echo "     This creates a .nemo archive optimized for 8×H100."
            echo ""

            # Check if we're in CPU-only mode
            if [ "${CPU_ONLY_MODE:-false}" = "true" ]; then
                echo "Warning: CPU_ONLY_MODE is enabled, but NeMo import may require GPU."
                echo "Skipping import. Please run without --smoke-test on a GPU server."
            else
                # Create parent directory
                mkdir -p "$(dirname "$MODEL_NEMO_PATH")"

                # Note: For multi-GPU import, use torchrun:
                # torchrun --nproc_per_node=8 scripts/convert/import_to_nemo.py ...
                # For now, using single process (slower but works on single GPU)
                python3 scripts/convert/import_to_nemo.py \
                    --bf16-dir "$MODEL_BF16_DIR" \
                    --output "$MODEL_NEMO_PATH" \
                    --tensor-parallel 8 \
                    --pipeline-parallel 1

                echo "✓ NeMo import complete"
            fi
        else
            echo "0.3: NeMo import already complete"
            echo "     ✓ NeMo model exists at: $MODEL_NEMO_PATH"
            echo "     To re-import: rm -f $MODEL_NEMO_PATH"
        fi

        echo ""
        echo "✓ Model setup complete"
        echo "  NeMo checkpoint: $MODEL_NEMO_PATH"
    fi
else
    echo "Skipping model setup (--skip-model-setup)"
    echo "Note: Training will fail if NeMo model doesn't exist at:"
    echo "      ${MODEL_NEMO_PATH:-/data/models/deepseek-v3-base_tp8_pp1.nemo}"
fi

# Step 1: Data preparation
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 1: Data Preparation"
    echo "========================================="
    
    # Define data paths
    RAW_XML_DIR=${RAW_XML_DIR:-data/raw_xml}
    JSONL_OUTPUT=${JSONL_OUTPUT:-data/jsonl/all.jsonl}
    HF_DATASET_DIR=${HF_DATASET_DIR:-data/hf_datasets/sft_dataset}
    NEMO_DATASET_DIR=${NEMO_DATASET_DIR:-data/nemo/sft_dataset}
    
    # Create output directories
    mkdir -p data/jsonl
    mkdir -p data/hf_datasets
    mkdir -p data/nemo
    
    # 1.0: Unpack raw XML if needed
    if [ ! -d "$RAW_XML_DIR" ] || [ -z "$(ls -A $RAW_XML_DIR 2>/dev/null)" ]; then
        if [ -f "data/raw_xml.zip" ]; then
            echo ""
            echo "1.0: Unpacking raw XML data..."
            mkdir -p "$RAW_XML_DIR"
            unzip -q data/raw_xml.zip -d data/
            echo "✓ Raw XML data unpacked to $RAW_XML_DIR"
        else
            echo "Error: No XML data found!"
            echo "  - Expected: $RAW_XML_DIR/ (directory with XML files)"
            echo "  - Or: data/raw_xml.zip (archive to unpack)"
            echo ""
            echo "Please place your XML files in $RAW_XML_DIR/ and re-run."
            exit 1
        fi
    else
        echo ""
        echo "1.0: Raw XML data already available"
        echo "     ✓ Found XML files in: $RAW_XML_DIR"
        XML_COUNT=$(find "$RAW_XML_DIR" -name "*.xml" | wc -l)
        echo "     ✓ XML file count: $XML_COUNT"
    fi
    
    # 1.1: XML to JSONL
    echo ""
    echo "1.1: Converting XML to JSONL..."
    if [ ! -f "$JSONL_OUTPUT" ]; then
        python3 src/parsers/xml_to_jsonl.py \
            --input_dir "$RAW_XML_DIR" \
            --output_file "$JSONL_OUTPUT"
        echo "✓ JSONL created: $JSONL_OUTPUT"
    else
        echo "✓ JSONL already exists: $JSONL_OUTPUT"
        RECORD_COUNT=$(wc -l < "$JSONL_OUTPUT")
        echo "  Record count: $RECORD_COUNT"
    fi
    
    # 1.2: JSONL to HF Dataset
    echo ""
    echo "1.2: Creating HuggingFace Dataset..."
    if [ ! -d "$HF_DATASET_DIR" ] || [ ! -f "$HF_DATASET_DIR/dataset_info.json" ]; then
        python3 src/data/convert_dataset.py \
            --jsonl "$JSONL_OUTPUT" \
            --out_dir "$HF_DATASET_DIR" \
            --train_end ${TRAIN_END_DATE:-2024-12-31} \
            --test_start ${TEST_START_DATE:-2025-01-01} \
            --validation_days ${VALIDATION_DAYS:-30}
        echo "✓ HuggingFace dataset created: $HF_DATASET_DIR"
    else
        echo "✓ HuggingFace dataset already exists: $HF_DATASET_DIR"
    fi
    
    # 1.3: Export NeMo dataset
    echo ""
    echo "1.3: Exporting NeMo dataset..."
    if [ ! -d "$NEMO_DATASET_DIR" ] || [ ! -f "$NEMO_DATASET_DIR/training.jsonl" ]; then
        python3 src/data/export_nemo_dataset.py \
            --dataset_dir "$HF_DATASET_DIR" \
            --output_dir "$NEMO_DATASET_DIR" \
            --template ${NEMO_TEMPLATE:-chatml} \
            --include_metadata
        echo "✓ NeMo dataset exported: $NEMO_DATASET_DIR"
    else
        echo "✓ NeMo dataset already exists: $NEMO_DATASET_DIR"
        for split in training validation test; do
            if [ -f "$NEMO_DATASET_DIR/$split.jsonl" ]; then
                COUNT=$(wc -l < "$NEMO_DATASET_DIR/$split.jsonl")
                echo "  - $split.jsonl: $COUNT records"
            fi
        done
    fi
    
    echo ""
    echo "✓ Data preparation complete"
    echo "  JSONL: $JSONL_OUTPUT"
    echo "  HF Dataset: $HF_DATASET_DIR"
    echo "  NeMo Dataset: $NEMO_DATASET_DIR"
else
    echo "Skipping data preparation (--skip-data)"
    echo "Note: Training requires NeMo dataset at: ${NEMO_DATASET_DIR:-data/nemo/sft_dataset}"
fi

# Step 2: Training
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 2: SFT Training"
    echo "========================================="

    CONFIG_FILE=${CONFIG_FILE:-configs/nemo/finetune.yaml}
    OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/nemo_runs/latest}
    
    # Validate config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Training config not found: $CONFIG_FILE"
        exit 1
    fi
    
    # Validate dataset exists
    NEMO_DATASET_DIR=${NEMO_DATASET_DIR:-data/nemo/sft_dataset}
    if [ ! -d "$NEMO_DATASET_DIR" ] || [ ! -f "$NEMO_DATASET_DIR/training.jsonl" ]; then
        echo "Error: NeMo dataset not found at: $NEMO_DATASET_DIR"
        echo "Please run data preparation first (without --skip-data)"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs

    if [ "$SMOKE_TEST" = true ]; then
        echo "Running NeMo smoke test (10 steps)..."
        python3 src/train/train_nemo.py \
            --config "$CONFIG_FILE" \
            --output "$OUTPUT_DIR" \
            --smoke-test
    else
        echo "Running NeMo full training..."
        echo "Config: $CONFIG_FILE"
        echo "Output: $OUTPUT_DIR"
        echo ""
        python3 src/train/train_nemo.py \
            --config "$CONFIG_FILE" \
            --output "$OUTPUT_DIR"
    fi

    echo ""
    echo "✓ Training complete"
    echo "  Output directory: $OUTPUT_DIR"
    if [ -f "$OUTPUT_DIR/deepseek_v3_finetune.nemo" ]; then
        echo "  ✓ Checkpoint: $OUTPUT_DIR/deepseek_v3_finetune.nemo"
    fi
else
    echo "Skipping training (--skip-train)"
    echo "Note: Evaluation requires trained model at: ${OUTPUT_DIR:-checkpoints/nemo_runs/latest}/deepseek_v3_finetune.nemo"
fi

# Step 3: Evaluation
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 3: Evaluation & Backtesting"
    echo "========================================="
    
    OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/nemo_runs/latest}
    NEMO_DATASET_DIR=${NEMO_DATASET_DIR:-data/nemo/sft_dataset}
    EVAL_RESULTS_CSV=${EVAL_RESULTS_CSV:-results/eval_results.csv}
    BACKTEST_CONFIG=${BACKTEST_CONFIG:-configs/backtest_config.yaml}
    BACKTEST_OUTPUT=${BACKTEST_OUTPUT:-backtests/baseline.csv}
    
    # Validate model exists
    TRAINED_MODEL="$OUTPUT_DIR/deepseek_v3_finetune.nemo"
    if [ ! -f "$TRAINED_MODEL" ]; then
        echo "Error: Trained model not found: $TRAINED_MODEL"
        echo "Please run training first (without --skip-train)"
        exit 1
    fi
    
    # Validate dataset exists
    if [ ! -d "$NEMO_DATASET_DIR" ]; then
        echo "Error: NeMo dataset not found: $NEMO_DATASET_DIR"
        echo "Please run data preparation first (without --skip-data)"
        exit 1
    fi
    
    # Create output directories
    mkdir -p results
    mkdir -p backtests
    
    # 3.1: Model evaluation
    echo ""
    echo "3.1: Evaluating model predictions..."
    echo "Model: $TRAINED_MODEL"
    echo "Dataset: $NEMO_DATASET_DIR"
    echo "Output: $EVAL_RESULTS_CSV"
    echo ""
    
    python3 src/eval/evaluate_nemo.py \
        --model "$TRAINED_MODEL" \
        --dataset "$NEMO_DATASET_DIR" \
        --results "$EVAL_RESULTS_CSV" \
        --split ${EVAL_SPLIT:-test}
    
    echo "✓ Evaluation complete: $EVAL_RESULTS_CSV"
    
    # Check if metrics JSON was created
    if [ -f "${EVAL_RESULTS_CSV}.metrics.json" ]; then
        echo "✓ Metrics saved: ${EVAL_RESULTS_CSV}.metrics.json"
    fi
    
    # 3.2: Backtest
    echo ""
    echo "3.2: Running backtest simulation..."
    echo "Evaluation results: $EVAL_RESULTS_CSV"
    echo "Config: $BACKTEST_CONFIG"
    echo "Output: $BACKTEST_OUTPUT"
    echo ""
    
    if [ ! -f "$BACKTEST_CONFIG" ]; then
        echo "Error: Backtest config not found: $BACKTEST_CONFIG"
        exit 1
    fi
    
    # Note: Using --eval_csv because results are in CSV format
    python3 src/backtest/trading_backtest.py \
        --eval_csv "$EVAL_RESULTS_CSV" \
        --config "$BACKTEST_CONFIG" \
        --out "$BACKTEST_OUTPUT"
    
    echo "✓ Backtest complete: $BACKTEST_OUTPUT"
    
    # Check if backtest metrics JSON was created
    if [ -f "${BACKTEST_OUTPUT%.*}_metrics.json" ]; then
        echo "✓ Backtest metrics: ${BACKTEST_OUTPUT%.*}_metrics.json"
    fi
    
    echo ""
    echo "✓ Evaluation & backtesting complete"
else
    echo "Skipping evaluation (--skip-eval)"
fi

# Summary
echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "End time: $(date)"
echo ""

if [ "$SKIP_MODEL_SETUP" = false ]; then
    echo "Model Setup:"
    echo "  - Base Model: ${MODEL_NEMO_PATH:-/data/models/deepseek-v3-base_tp8_pp1.nemo}"
    echo ""
fi

if [ "$SKIP_DATA" = false ]; then
    echo "Data:"
    echo "  - JSONL: ${JSONL_OUTPUT:-data/jsonl/all.jsonl}"
    echo "  - HF Dataset: ${HF_DATASET_DIR:-data/hf_datasets/sft_dataset}"
    echo "  - NeMo Dataset: ${NEMO_DATASET_DIR:-data/nemo/sft_dataset}"
    echo ""
fi

if [ "$SKIP_TRAIN" = false ]; then
    echo "Training:"
    echo "  - Output Dir: ${OUTPUT_DIR:-checkpoints/nemo_runs/latest}"
    echo "  - Checkpoint: ${OUTPUT_DIR:-checkpoints/nemo_runs/latest}/deepseek_v3_finetune.nemo"
    echo ""
fi

if [ "$SKIP_EVAL" = false ]; then
    echo "Evaluation:"
    echo "  - Results: ${EVAL_RESULTS_CSV:-results/eval_results.csv}"
    echo "  - Backtest: ${BACKTEST_OUTPUT:-backtests/baseline.csv}"
    if [ -f "${EVAL_RESULTS_CSV:-results/eval_results.csv}.metrics.json" ]; then
        echo "  - Metrics: ${EVAL_RESULTS_CSV:-results/eval_results.csv}.metrics.json"
    fi
    echo ""
fi

echo "Logs: ${LOG_DIR:-logs}/"
echo ""
echo "========================================="
echo "All stages completed successfully!"
echo "========================================="
echo ""
