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
SKIP_DATA=false
SKIP_TRAIN=false
SKIP_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke-test)
            SMOKE_TEST=true
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
            echo "Usage: $0 [--smoke-test] [--skip-data] [--skip-train] [--skip-eval]"
            exit 1
            ;;
    esac
done

if [ "$SMOKE_TEST" = true ]; then
    echo "SMOKE TEST MODE"
    export CPU_ONLY_MODE=true
    export SMOKE_TEST_MODE=true
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
    
    # 1.3: Tokenization
    echo ""
    echo "1.3: Tokenizing dataset..."
    python3 src/data/tokenize_and_shard.py \
        --dataset_dir ${HF_DATASET_DIR:-data/hf_datasets/sft_dataset} \
        --tokenizer ${BASE_MODEL:-deepseek-ai/DeepSeek-V3.2-Exp} \
        --max_length ${MAX_LENGTH:-2048}
    
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
    
    CONFIG_FILE="configs/sft_config.yaml"
    
    if [ "$SMOKE_TEST" = true ]; then
        echo "Running smoke test (10 steps)..."
        python3 src/train/train_sft.py \
            --config $CONFIG_FILE \
            --smoke_test
    else
        echo "Running full training..."
        python3 src/train/train_sft.py \
            --config $CONFIG_FILE
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
    
    MODEL_DIR=${OUTPUT_DIR:-checkpoints/sft-deepseek-v3.2exp-longctx}
    
    # 3.1: Model evaluation
    echo ""
    echo "3.1: Evaluating model predictions..."
    python3 src/eval/evaluate_sft.py \
        --model_dir $MODEL_DIR \
        --dataset_dir ${HF_DATASET_DIR:-data/hf_datasets/sft_dataset} \
        --out ${EVAL_RESULTS_CSV:-results/eval_results.csv} \
        --forward_days 5 \
        --split test
    
    # 3.2: Backtest
    echo ""
    echo "3.2: Running backtest simulation..."
    python3 src/backtest/trading_backtest.py \
        --eval_csv ${EVAL_RESULTS_CSV:-results/eval_results.csv} \
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
echo "Results:"
echo "  - Model: ${OUTPUT_DIR:-checkpoints/sft-deepseek-v3.2exp-longctx}"
echo "  - Evaluation: ${EVAL_RESULTS_CSV:-results/eval_results.csv}"
echo "  - Backtest: backtests/baseline.csv"
echo ""
echo "Check logs in: ${LOG_DIR:-logs}/"
echo ""
