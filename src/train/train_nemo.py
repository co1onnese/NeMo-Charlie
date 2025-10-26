#!/usr/bin/env python3
"""NeMo-based fine-tuning entrypoint for DeepSeek-V3."""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

try:
    from nemo.collections import llm
    from nemo.collections.llm.gpt.data.fine_tuning import FineTuningDataModule
    from nemo.collections.llm.tokenizers import AutoTokenizer as NemoAutoTokenizer
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "NeMo toolkit is required. Install dependencies with INSTALL_NEMO=true scripts/setup_env.sh"
    ) from exc

from src.utils.logger import setup_logger
from src.utils.manifest import create_manifest


load_dotenv()
LOGGER = setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek-V3 using NeMo")
    parser.add_argument("--config", required=True, help="Path to NeMo config JSON/YAML")
    parser.add_argument("--resume", default=None, help="Optional NeMo checkpoint to resume from")
    parser.add_argument("--output", default="checkpoints/nemo_runs", help="Run output directory")
    parser.add_argument("--smoke-test", action="store_true", help="Short run for validation")
    return parser.parse_args()


def load_config(path: str) -> dict:
    ext = Path(path).suffix
    with open(path, "r", encoding="utf-8") as handle:
        if ext in {".json"}:
            return json.load(handle)
        import yaml

        return yaml.safe_load(handle)


def build_data_module(config: dict) -> FineTuningDataModule:
    dataset_root = Path(config["dataset"]["path"])
    tokenizer_name = config["dataset"].get("tokenizer", config["recipe"].get("resume_path"))

    LOGGER.info("Loading NeMo tokenizer from %s", tokenizer_name)
    tokenizer = NemoAutoTokenizer.from_pretrained(tokenizer_name)

    seq_length = config["train"].get("seq_length", 2048)
    micro_batch = config["train"].get("micro_batch_size", 1)
    global_batch = config["train"].get("global_batch_size", micro_batch)

    dataset_kwargs = {
        key: config["dataset"][key]
        for key in (
            "label_key",
            "answer_only_loss",
            "prompt_template",
            "truncation_field",
            "max_num_samples",
        )
        if key in config["dataset"]
    }
    if config["dataset"].get("packed_sequence_size"):
        from nemo.collections.llm.gpt.data.packed_sequence import PackedSequenceSpecs

        packed_specs = PackedSequenceSpecs(
            packed_sequence_size=config["dataset"]["packed_sequence_size"],
            pad_cu_seqlens=config["dataset"].get("pad_cu_seqlens", False),
        )
    else:
        packed_specs = None

    data_module = FineTuningDataModule(
        dataset_root=dataset_root,
        seq_length=seq_length,
        tokenizer=tokenizer,
        micro_batch_size=micro_batch,
        global_batch_size=global_batch,
        num_workers=config["dataset"].get("num_workers", 4),
        packed_sequence_specs=packed_specs,
        dataset_kwargs=dataset_kwargs,
    )

    data_module.prepare_data()
    data_module.setup("fit")
    return data_module


def run_training(config: dict, resume_path: Optional[str], output_dir: Path, smoke_test: bool) -> None:
    recipe_factory = config["recipe"].get("factory", "deepseek_v3")
    if recipe_factory != "deepseek_v3":
        raise ValueError("Currently only deepseek_v3 factory is supported")

    LOGGER.info("Initializing NeMo DeepSeek-V3 finetune recipe")
    recipe = llm.recipes.deepseek_v3.finetune_recipe(
        dir=str(output_dir),
        resume_path=resume_path or config["recipe"].get("resume_path"),
        name=config["recipe"].get("name", "deepseek_v3_finetune"),
        num_nodes=config["train"].get("num_nodes", 1),
        num_gpus_per_node=config["train"].get("gpus_per_node", 8),
        peft_scheme=config["train"].get("peft", "lora"),
        seq_length=config["train"].get("seq_length", 2048),
        performance_mode=config["train"].get("performance_mode", False),
    )

    data_module = build_data_module(config)
    recipe.data = data_module

    if smoke_test:
        LOGGER.warning("Running smoke test mode: limiting steps")
        recipe.trainer.max_steps = config["train"].get("smoke_test_steps", 10)
        recipe.trainer.limit_val_batches = 0
        recipe.log.ckpt.save_top_k = 0

    manifest_path = output_dir / "manifest.json"
    create_manifest(
        run_name=config["recipe"].get("name", "deepseek_v3_finetune"),
        config=config,
        output_path=str(manifest_path),
    )

    LOGGER.info("Starting training run via NeMo recipe")
    recipe()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_training(config, args.resume, output_dir, args.smoke_test)


if __name__ == "__main__":
    main()

