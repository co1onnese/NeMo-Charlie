#!/usr/bin/env python3
"""Export HuggingFace dataset splits into NeMo-compatible JSONL files."""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, Iterable, Optional

from datasets import Dataset, DatasetDict, load_from_disk
from tqdm import tqdm

try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover - handled via setup script
    raise SystemExit(
        "transformers is required. Install dependencies with scripts/setup_env.sh"
    ) from exc


LOGGER = logging.getLogger(__name__)

SPECIAL_TOKENS = [
    "<reasoning>",
    "</reasoning>",
    "<support>",
    "</support>",
    "<action>",
    "</action>",
    "<STRONG_BUY>",
    "<BUY>",
    "<HOLD>",
    "<SELL>",
    "<STRONG_SELL>",
]

SPLIT_NAME_MAP = {
    "train": "training",
    "validation": "validation",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HF dataset to NeMo JSONL format")
    parser.add_argument("--dataset_dir", required=True, help="Path to HF dataset (load_from_disk)")
    parser.add_argument("--output_dir", required=True, help="Destination directory for NeMo files")
    parser.add_argument(
        "--template",
        default="chatml",
        choices=["alpaca", "chatml", "simple"],
        help="Prompt template style",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Base tokenizer to extend with special tokens (HuggingFace identifier or path)",
    )
    parser.add_argument(
        "--tokenizer_out",
        default=None,
        help="Directory to write extended tokenizer (required if --tokenizer is set)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples per split (for smoke tests)",
    )
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="Include auxiliary fields under 'metadata' in each JSON line",
    )
    return parser.parse_args()


def extend_tokenizer(base_tokenizer: str, output_dir: Path) -> Path:
    LOGGER.info("Extending tokenizer %s with %d special tokens", base_tokenizer, len(SPECIAL_TOKENS))
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, trust_remote_code=True)

    num_added = tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    LOGGER.info("Added %d tokens; vocab size is now %d", num_added, len(tokenizer))

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Extended tokenizer saved to %s", output_dir)
    return output_dir


def format_prompt(example: Dict[str, str], template: str) -> Dict[str, str]:
    instruction = example.get("instruction", "") or ""
    input_text = example.get("input", "") or ""
    output_text = example.get("output", "") or ""

    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    elif template == "chatml":
        if input_text:
            user_segment = f"<|User|>: {instruction}\n{input_text}"
        else:
            user_segment = f"<|User|>: {instruction}"
        prompt = f"{user_segment}<|Assistant|>: "
    else:  # simple
        prompt = f"{instruction}\n{input_text}\n"

    return {"input": prompt, "output": output_text}


def sanitize_value(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return str(value)


def export_split(
    dataset: Dataset,
    output_path: Path,
    template: str,
    max_samples: Optional[int],
    include_metadata: bool,
) -> int:
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in tqdm(dataset, desc=f"Writing {output_path.name}"):
            if max_samples is not None and count >= max_samples:
                break

            formatted = format_prompt(example, template)
            record = {
                "input": formatted["input"],
                "output": formatted["output"],
            }

            if include_metadata:
                metadata = {
                    key: sanitize_value(value)
                    for key, value in example.items()
                    if key not in {"instruction", "input", "output"}
                }
                metadata = {k: v for k, v in metadata.items() if v is not None}
                if metadata:
                    record["metadata"] = metadata

            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dataset = load_from_disk(args.dataset_dir)
    if not isinstance(dataset, (Dataset, DatasetDict)):
        raise SystemExit("Loaded object is not a Dataset or DatasetDict")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = None
    if args.tokenizer:
        if not args.tokenizer_out:
            raise SystemExit("--tokenizer-out is required when --tokenizer is provided")
        tokenizer_path = extend_tokenizer(args.tokenizer, Path(args.tokenizer_out))

    stats = {}
    if isinstance(dataset, Dataset):
        # single split only
        split_name = SPLIT_NAME_MAP.get(dataset.info.split, "training")
        stats[split_name] = export_split(
            dataset,
            output_dir / f"{split_name}.jsonl",
            args.template,
            args.max_samples,
            args.include_metadata,
        )
    else:
        for split, nemo_name in SPLIT_NAME_MAP.items():
            if split not in dataset:
                LOGGER.warning("Split '%s' not found in dataset; skipping", split)
                continue
            stats[nemo_name] = export_split(
                dataset[split],
                output_dir / f"{nemo_name}.jsonl",
                args.template,
                args.max_samples,
                args.include_metadata,
            )

    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump({"splits": stats, "template": args.template}, handle, indent=2)
    LOGGER.info("Wrote dataset statistics to %s", stats_path)

    if tokenizer_path:
        LOGGER.info("Extended tokenizer available at %s", tokenizer_path)


if __name__ == "__main__":
    main()

