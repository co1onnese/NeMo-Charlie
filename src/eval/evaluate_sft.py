#!/usr/bin/env python3
"""
evaluate_sft.py
Evaluate a fine-tuned SFT model on both NLP and market outcomes.
Uses eodhd.com API for price data with yfinance fallback.

Usage:
  python src/eval/evaluate_sft.py --model_dir checkpoints/sft-run \
    --dataset_dir data/hf_datasets/sft_dataset \
    --out results/eval_results.csv
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger
from src.utils.eval_utils import extract_action
from src.data.price_data import PriceDataClient

# Load environment
load_dotenv()

logger = setup_logger(__name__)

ACTION_DIRECTION = {
    "STRONG_BUY": +1,
    "BUY": +1,
    "HOLD": 0,
    "SELL": -1,
    "STRONG_SELL": -1
}


def generate_prediction(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0
) -> str:
    """Generate prediction from model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    
    return text


def format_prompt(example: dict) -> str:
    """Format example as prompt."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    return prompt


def evaluate_classification(df: pd.DataFrame) -> dict:
    """Compute classification metrics."""
    mask = df["gt_action"].notna() & df["pred_action"].notna()
    mask &= (df["gt_action"] != "UNKNOWN") & (df["pred_action"] != "UNKNOWN")
    
    if mask.sum() == 0:
        return {"error": "No valid predictions for classification"}
    
    y_true = df.loc[mask, "gt_action"].tolist()
    y_pred = df.loc[mask, "pred_action"].tolist()
    
    # Get unique labels
    labels = sorted(list(set(y_true + y_pred)))
    
    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "accuracy": report.get("accuracy", 0.0),
        "num_samples": mask.sum()
    }


def evaluate_financial(df: pd.DataFrame, forward_days: int) -> dict:
    """Compute financial metrics."""
    valid = df.dropna(subset=["realized_return", "pred_action"])
    valid = valid[valid["pred_action"] != "UNKNOWN"]
    
    if len(valid) == 0:
        return {"error": "No valid realized returns"}
    
    # Direction correctness
    valid["direction_correct"] = valid.apply(
        lambda row: (
            (np.sign(row["realized_return"]) == ACTION_DIRECTION.get(row["pred_action"], 0))
            if ACTION_DIRECTION.get(row["pred_action"], 0) != 0
            else abs(row["realized_return"]) < 0.001
        ),
        axis=1
    )
    
    hit_rate = valid["direction_correct"].mean()
    mean_return = valid["realized_return"].mean()
    std_return = valid["realized_return"].std()
    
    # Approximate annualized Sharpe
    if std_return > 0 and not np.isnan(std_return):
        sharpe = (mean_return / std_return) * np.sqrt(252 / forward_days)
    else:
        sharpe = np.nan
    
    # Per-action statistics
    action_stats = {}
    for action in ACTION_DIRECTION.keys():
        action_df = valid[valid["pred_action"] == action]
        if len(action_df) > 0:
            action_stats[action] = {
                "count": len(action_df),
                "mean_return": action_df["realized_return"].mean(),
                "hit_rate": action_df["direction_correct"].mean() if "direction_correct" in action_df else np.nan
            }
    
    return {
        "hit_rate": hit_rate,
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe_ratio": sharpe,
        "num_samples": len(valid),
        "per_action_stats": action_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model")
    parser.add_argument("--model_dir", required=True, help="Path to trained model")
    parser.add_argument("--dataset_dir", default=None, help="Path to HF dataset (default from .env)")
    parser.add_argument("--out", default=None, help="Output CSV path (default from .env)")
    parser.add_argument("--forward_days", type=int, default=5, help="Forward return window")
    parser.add_argument("--split", default="test", choices=["test", "validation", "train"], help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU evaluation")
    
    args = parser.parse_args()
    
    # Get defaults from environment
    if args.dataset_dir is None:
        args.dataset_dir = os.getenv("HF_DATASET_DIR", "data/hf_datasets/sft_dataset")
    if args.out is None:
        args.out = os.getenv("EVAL_RESULTS_CSV", "results/eval_results.csv")
    
    logger.info("=" * 80)
    logger.info("SFT Model Evaluation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_dir}")
    logger.info(f"Dataset: {args.dataset_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Forward days: {args.forward_days}")
    
    # Setup device
    if args.cpu_only or not torch.cuda.is_available():
        device = "cpu"
        logger.warning("Using CPU for evaluation")
    else:
        device = "cuda"
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = load_from_disk(args.dataset_dir)
        if args.split not in dataset:
            logger.error(f"Split '{args.split}' not found in dataset. Available: {list(dataset.keys())}")
            sys.exit(1)
        eval_dataset = dataset[args.split]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Limit samples if requested
    if args.max_samples and args.max_samples < len(eval_dataset):
        logger.info(f"Limiting to {args.max_samples} samples")
        eval_dataset = eval_dataset.select(range(args.max_samples))
    
    logger.info(f"Evaluating {len(eval_dataset)} samples...")
    
    # Initialize price data client
    price_client = PriceDataClient()
    
    # Evaluate samples
    records = []
    for example in tqdm(eval_dataset, desc="Evaluating"):
        ticker = example.get("ticker")
        as_of_date = example.get("as_of_date")
        
        # Generate prediction
        prompt = format_prompt(example)
        try:
            completion = generate_prediction(model, tokenizer, prompt, device)
        except Exception as e:
            logger.warning(f"Generation failed for {ticker} {as_of_date}: {e}")
            completion = ""
        
        # Extract actions
        pred_action = extract_action(completion)
        gt_action = extract_action(example.get("output", ""))
        
        # Get forward return
        realized_return = None
        if ticker and as_of_date:
            realized_return = price_client.get_forward_return(
                ticker, as_of_date, args.forward_days
            )
        
        records.append({
            "uid": example.get("uid"),
            "ticker": ticker,
            "as_of_date": as_of_date,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "realized_return": realized_return,
            "prompt": prompt,
            "completion": completion,
            "gt_output": example.get("output", "")
        })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info(f"Saved results to: {args.out}")
    
    # Compute metrics
    logger.info("\n" + "=" * 80)
    logger.info("CLASSIFICATION METRICS")
    logger.info("=" * 80)
    
    class_metrics = evaluate_classification(df)
    if "error" in class_metrics:
        logger.warning(class_metrics["error"])
    else:
        logger.info(f"Accuracy: {class_metrics['accuracy']:.4f}")
        logger.info(f"Samples: {class_metrics['num_samples']}")
        logger.info("\nPer-class metrics:")
        for label in class_metrics["labels"]:
            if label in class_metrics["report"]:
                metrics = class_metrics["report"][label]
                logger.info(f"  {label:15s} - P: {metrics['precision']:.3f}  R: {metrics['recall']:.3f}  F1: {metrics['f1-score']:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("FINANCIAL METRICS")
    logger.info("=" * 80)
    
    fin_metrics = evaluate_financial(df, args.forward_days)
    if "error" in fin_metrics:
        logger.warning(fin_metrics["error"])
    else:
        logger.info(f"Hit Rate: {fin_metrics['hit_rate']:.4f}")
        logger.info(f"Mean Return: {fin_metrics['mean_return']:.6f}")
        logger.info(f"Std Return: {fin_metrics['std_return']:.6f}")
        logger.info(f"Sharpe Ratio (est): {fin_metrics['sharpe_ratio']:.3f}")
        logger.info(f"Samples: {fin_metrics['num_samples']}")
        
        logger.info("\nPer-action statistics:")
        for action, stats in fin_metrics["per_action_stats"].items():
            logger.info(f"  {action:15s} - Count: {stats['count']:4d}  Mean Return: {stats['mean_return']:+.6f}  Hit Rate: {stats['hit_rate']:.3f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
