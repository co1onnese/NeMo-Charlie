#!/usr/bin/env python3
"""Evaluate NeMo-trained DeepSeek-V3 models on trading dataset."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

try:
    from nemo.collections.llm.inference.base import setup_model_and_tokenizer, generate
    from nemo.lightning import Trainer
    from nemo.lightning.pytorch.strategies import MegatronStrategy
    from megatron.core.inference.common_inference_params import CommonInferenceParams
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "NeMo toolkit is required. Install dependencies with INSTALL_NEMO=true scripts/setup_env.sh"
    ) from exc

from src.utils.eval_utils import extract_action, load_metrics_config
from src.eval.metrics import compute_classification_metrics, compute_financial_metrics
from src.utils.logger import setup_logger
from src.data.price_data import PriceDataClient


LOGGER = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NeMo DeepSeek-V3 model")
    parser.add_argument("--model", required=True, help="Path to .nemo checkpoint")
    parser.add_argument("--dataset", required=True, help="Path to NeMo JSONL dataset directory")
    parser.add_argument("--results", default="results/eval_results.csv", help="CSV output path")
    parser.add_argument("--metrics-json", default=None, help="Optional path to save metrics JSON")
    parser.add_argument("--split", default="validation", choices=["training", "validation", "test"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional maximum records to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--forward-days", type=int, default=5, help="Forward return window for price data")
    return parser.parse_args()


def load_dataset(split_path: Path, limit: Optional[int]) -> pd.DataFrame:
    records = []
    with split_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            records.append(json.loads(line))
    return pd.DataFrame(records)


def evaluate(args: argparse.Namespace) -> pd.DataFrame:
    dataset_dir = Path(args.dataset)
    split_file = dataset_dir / f"{args.split}.jsonl"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    LOGGER.info("Setting up NeMo inference from %s", args.model)
    
    # Configure trainer for inference
    strategy = MegatronStrategy(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        ddp="pytorch",
    )
    trainer = Trainer(
        devices=8,
        accelerator="gpu",
        strategy=strategy,
    )
    
    # Setup model and tokenizer for inference
    inference_model, tokenizer = setup_model_and_tokenizer(
        path=Path(args.model),
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_max_seq_length=args.max_new_tokens + 2048,
    )

    df = load_dataset(split_file, args.max_samples)
    LOGGER.info("Evaluating %d samples from %s", len(df), split_file.name)

    price_client = PriceDataClient()
    outputs = []
    
    inference_params = CommonInferenceParams(
        num_tokens_to_generate=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0 else 1.0,
        top_k=1 if args.temperature == 0 else 50,
        top_p=0.9 if args.temperature > 0 else 1.0,
    )

    for row in tqdm(df.itertuples(), total=len(df), desc="Generating"):
        prompt = row.input
        
        # Generate using NeMo inference
        result = generate(
            model=inference_model,
            tokenizer=tokenizer,
            prompts=[prompt],
            max_batch_size=1,
            inference_params=inference_params,
        )
        completion = result["sentences"][0] if result and "sentences" in result else ""

        pred_action = extract_action(completion)
        ticker = row.metadata.get("ticker") if isinstance(row.metadata, dict) else None
        as_of_date = row.metadata.get("as_of_date") if isinstance(row.metadata, dict) else None
        realized_return = None
        if ticker and as_of_date:
            realized_return = price_client.get_forward_return(ticker, as_of_date, args.forward_days)

        outputs.append(
            {
                "prompt": prompt,
                "completion": completion,
                "gt_output": row.output,
                "pred_action": pred_action,
                "gt_action": extract_action(row.output or ""),
                "ticker": ticker,
                "as_of_date": as_of_date,
                "realized_return": realized_return,
            }
        )

    results_df = pd.DataFrame(outputs)
    results_dir = Path(args.results).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.results, index=False)
    LOGGER.info("Saved evaluation results to %s", args.results)
    metrics_config = load_metrics_config()
    class_metrics = compute_classification_metrics(results_df, metrics_config.classification)
    fin_metrics = compute_financial_metrics(results_df, metrics_config.financial)

    LOGGER.info("Classification metrics: accuracy=%.4f, samples=%d", class_metrics.accuracy, class_metrics.samples)
    LOGGER.info(
        "Financial metrics: hit_rate=%.4f, sharpe=%.3f, samples=%d",
        fin_metrics.hit_rate,
        fin_metrics.sharpe_ratio,
        fin_metrics.samples,
    )

    metrics_payload = {
        "classification": {
            "accuracy": class_metrics.accuracy,
            "samples": class_metrics.samples,
            "report": class_metrics.report,
        },
        "financial": {
            "hit_rate": fin_metrics.hit_rate,
            "mean_return": fin_metrics.mean_return,
            "std_return": fin_metrics.std_return,
            "sharpe_ratio": fin_metrics.sharpe_ratio,
            "samples": fin_metrics.samples,
        },
    }

    metrics_path = Path(args.metrics_json) if args.metrics_json else Path(args.results + ".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    LOGGER.info("Metrics JSON saved to %s", metrics_path)

    return results_df


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()

