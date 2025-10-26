#!/usr/bin/env python3
"""
evaluate_sft.py
Evaluate a fine-tuned SFT model on both NLP and market outcomes.
Produces a per-sample CSV and prints aggregates.

Usage:
  python src/eval/evaluate_sft.py --model_dir checkpoints/sft-run --dataset_dir data/hf_datasets/sft_dataset --out results/eval_results.csv --forward_days 5
"""
import argparse
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import classification_report, confusion_matrix
from utils.eval_utils import extract_action, load_price_cache, get_forward_returns_for_sample

ACTION_DIR = {
    "STRONG_BUY": +1,
    "BUY": +1,
    "HOLD": 0,
    "SELL": -1,
    "STRONG_SELL": -1
}

def run_generation(model, tokenizer, prompt, device, max_new_tokens=256):
    inp = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--forward_days", type=int, default=5)
    p.add_argument("--price_cache", default=None, help="optional CSV/Parquet with cached price histories")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(device)

    ds = load_from_disk(args.dataset_dir)
    # choose test split if present otherwise use validation
    if "test" in ds:
        dataset = ds["test"]
    elif "validation" in ds:
        dataset = ds["validation"]
    else:
        dataset = ds["train"].train_test_split(test_size=0.05, seed=42)["test"]

    price_cache = load_price_cache(args.price_cache) if args.price_cache else {}

    records = []
    for ex in tqdm(dataset, desc="Eval"):
        ticker = ex.get("ticker") or ex.get("symbol") or None
        as_of = ex.get("as_of_date") or ex.get("date") or None
        # build prompt - same template as training
        prompt = f"### Instruction:\n{ex.get('instruction','')}\n\n### Input:\n{ex.get('input','')}\n\n### Response:\n"
        completion = run_generation(model, tokenizer, prompt, device, max_new_tokens=512)
        pred_action = extract_action(completion)
        gt_action = extract_action(ex.get("output","") or "")
        returns = get_forward_returns_for_sample(ticker, as_of, args.forward_days, price_cache)
        direction_correct = None
        if returns is not None and not np.isnan(returns):
            sign = np.sign(returns)
            dir_expected = ACTION_DIR.get(pred_action, 0)
            direction_correct = (sign == dir_expected) if dir_expected != 0 else (abs(returns) < 1e-9)
        records.append({
            "uid": ex.get("uid"),
            "ticker": ticker,
            "as_of_date": as_of,
            "gt_action": gt_action,
            "pred_action": pred_action,
            "realized_return": returns,
            "direction_correct": direction_correct,
            "prompt": prompt,
            "completion": completion
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")

    # Classification metrics (filter unknown)
    mask = df["gt_action"].notna() & (df["gt_action"].str.len() > 0)
    y_true = df.loc[mask, "gt_action"].fillna("UNKNOWN").tolist()
    y_pred = df.loc[mask, "pred_action"].fillna("UNKNOWN").tolist()
    print("Classification report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Financial aggregates
    valid = df.dropna(subset=["realized_return"])
    if not valid.empty:
        hit_rate = valid["direction_correct"].mean()
        mean_ret = valid["realized_return"].mean()
        vol = valid["realized_return"].std()
        sharpe = (mean_ret / vol) * np.sqrt(252 / args.forward_days) if vol and not np.isnan(vol) and vol > 0 else np.nan
        print(f"Hit rate: {hit_rate:.4f}")
        print(f"Mean realized return (raw): {mean_ret:.6f}")
        print(f"Approx annualized sharpe (est): {sharpe:.3f}")
    else:
        print("No valid realized returns available for the chosen forward_days.")

if __name__ == "__main__":
    main()
