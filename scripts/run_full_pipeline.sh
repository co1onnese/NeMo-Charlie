#!/bin/bash
# run_full_pipeline.sh
# Complete end-to-end pipeline for SFT training

set -e  # Exit on error

echo "========================================="
echo "SFT Trading Pipeline - Full Run"
echo "========================================="

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Activate venv
if [ -d "venv/bin" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

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
            echo "✓ DeepSeek-V3 source already exists at: $MODEL_SOURCE_DIR"
        fi

        echo ""

        # 0.2: Convert FP8 to BF16
        if [ ! -d "$MODEL_BF16_DIR" ] || [ ! -f "$MODEL_BF16_DIR/config.json" ]; then
            echo "0.2: Converting FP8 weights to BF16..."
            echo "     This requires GPU and may take 15-30 minutes."
            echo ""

            # Check if we're in CPU-only mode
            if [ "${CPU_ONLY_MODE:-false}" = "true" ]; then
                echo "Warning: CPU_ONLY_MODE is enabled, but conversion requires GPU."
                echo "Skipping conversion. Please run without --smoke-test on a GPU server."
            else
                bash scripts/convert/convert_deepseek_v3.sh \
                    --source "$MODEL_SOURCE_DIR" \
                    --output "$MODEL_BF16_DIR" \
                    --log logs/conversion/deepseek_v3_fp8_to_bf16.log

                echo "✓ BF16 conversion complete"
            fi
        else
            echo "✓ BF16 model already exists at: $MODEL_BF16_DIR"
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

                python3 scripts/convert/import_to_nemo.py \
                    --bf16-dir "$MODEL_BF16_DIR" \
                    --output "$MODEL_NEMO_PATH" \
                    --tensor-parallel 8 \
                    --pipeline-parallel 1 \
                    --sequence-length 131072 \
                    --model-name deepseek_v3

                echo "✓ NeMo import complete"
            fi
        else
            echo "✓ NeMo model already exists at: $MODEL_NEMO_PATH"
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
    
    # 1.1: XML to JSONL
    echo ""
    echo "1.1: Converting XML to JSONL..."
    python3 src/parsers/xml_to_jsonl.py \
        --input_dir ${RAW_XML_DIR:-data/raw_xml} \
        --output_file ${JSONL_OUTPUT:-data/jsonl/all.jsonl}
    
    # 1.2: JSONL to HF Dataset
    echo ""
    echo "1.2: Creating HuggingFace Dataset..."
    python3 src/data/convert_dataset.py \
        --jsonl ${JSONL_OUTPUT:-data/jsonl/all.jsonl} \
        --out_dir ${HF_DATASET_DIR:-data/hf_datasets/sft_dataset} \
        --train_end ${TRAIN_END_DATE:-2024-12-31} \
        --test_start ${TEST_START_DATE:-2025-01-01} \
        --validation_days ${VALIDATION_DAYS:-30}
    
    # 1.3: Export NeMo dataset
    echo ""
    echo "1.3: Exporting NeMo dataset..."
    python3 src/data/export_nemo_dataset.py \
        --dataset_dir ${HF_DATASET_DIR:-data/hf_datasets/sft_dataset} \
        --output_dir ${NEMO_DATASET_DIR:-data/nemo/sft_dataset} \
        --template ${NEMO_TEMPLATE:-chatml} \
        --include_metadata
    
    echo ""
    echo "✓ Data preparation complete"
else
    echo "Skipping data preparation (--skip-data)"
fi

# Step 2: Training
if [ "$SKIP_TRAIN" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 2: SFT Training"
    echo "========================================="

    CONFIG_FILE="configs/nemo/finetune.yaml"
    OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/nemo_runs/latest}
    mkdir -p "$(dirname "$OUTPUT_DIR")"

    if [ "$SMOKE_TEST" = true ]; then
        echo "Running NeMo smoke test..."
        python3 src/train/train_nemo.py \
            --config "$CONFIG_FILE" \
            --output "$OUTPUT_DIR" \
            --smoke-test
    else
        echo "Running NeMo full training via launcher..."
        python3 src/train/train_nemo.py \
            --config "$CONFIG_FILE" \
            --output "$OUTPUT_DIR"
    fi

    echo ""
    echo "✓ Training complete"
else
    echo "Skipping training (--skip-train)"
fi

# Step 3: Evaluation
if [ "$SKIP_EVAL" = false ]; then
    echo ""
    echo "========================================="
    echo "Step 3: Evaluation"
    echo "========================================="
    
    MODEL_DIR=${OUTPUT_DIR:-checkpoints/nemo_runs/main}
    
    # 3.1: Model evaluation
    echo ""
    echo "3.1: Evaluating model predictions..."
    python3 src/eval/evaluate_nemo.py \
        --model ${MODEL_DIR}/deepseek_v3_finetune.nemo \
        --dataset ${NEMO_DATASET_DIR:-data/nemo/sft_dataset} \
        --results ${EVAL_RESULTS_CSV:-results/eval_results.csv}
    
    # 3.2: Backtest
    echo ""
    echo "3.2: Running backtest simulation..."
    python3 src/backtest/trading_backtest.py \
        --eval_jsonl ${EVAL_RESULTS_CSV:-results/eval_results.csv} \
        --config configs/backtest_config.yaml \
        --out backtests/baseline.csv
    
    echo ""
    echo "✓ Evaluation complete"
else
    echo "Skipping evaluation (--skip-eval)"
fi

# Summary
echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo "Model Setup:"
echo "  - Base Model: ${MODEL_NEMO_PATH:-/data/models/deepseek-v3-base_tp8_pp1.nemo}"
echo ""
echo "Results:"
echo "  - Fine-tuned Model: ${OUTPUT_DIR:-checkpoints/nemo_runs/main}/deepseek_v3_finetune.nemo"
echo "  - Evaluation: ${EVAL_RESULTS_CSV:-results/eval_results.csv}"
echo "  - Backtest: backtests/baseline.csv"
echo "  - Metrics: results/eval_results.csv.metrics.json"
echo ""
echo "Check logs in: ${LOG_DIR:-logs}/"
echo ""
