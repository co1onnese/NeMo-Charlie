#!/usr/bin/env python3
"""Import converted DeepSeek-V3 BF16 weights into a NeMo .nemo archive.

This script assumes the FP8→BF16 conversion has already been performed using
`scripts/convert/fp8_cast_bf16.py`. It wraps the NeMo LLM import utility with
project-specific defaults (8-way tensor parallel for an 8×H100 topology).

Example:
    python scripts/convert/import_to_nemo.py \
        --bf16-dir checkpoints/bf16/deepseek-v3 \
        --output checkpoints/nemo/deepseek-v3-base_tp8_pp1.nemo

The script performs minimal validation of the input directory layout but does
not attempt to download or convert weights itself. Install NeMo and its
dependencies prior to running (see `requirements_nemo.txt`).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from nemo.collections import llm


def import_checkpoint(
    bf16_dir: Path,
    output_path: Path,
    tensor_parallel: int = 8,
    pipeline_parallel: int = 1,
    sequence_length: int = 131072,
    model_name: str = "deepseek_v3",
    extra_overrides: Optional[dict] = None,
) -> None:
    if not bf16_dir.exists():
        raise FileNotFoundError(f"BF16 directory not found: {bf16_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overrides = {
        "tensor_model_parallel_size": tensor_parallel,
        "pipeline_model_parallel_size": pipeline_parallel,
        "sequence_parallel": True,
        "max_position_embeddings": sequence_length,
    }
    if extra_overrides:
        overrides.update(extra_overrides)

    print("[INFO] Importing BF16 checkpoint into NeMo archive…")
    print(f"[INFO] BF16 source: {bf16_dir}")
    print(f"[INFO] Output archive: {output_path}")
    print(f"[INFO] Overrides: {json.dumps(overrides, indent=2)}")

    llm.import_ckpt(
        checkpoint_dir=str(bf16_dir),
        output_path=str(output_path),
        model_name=model_name,
        config_overrides=overrides,
    )

    print("[INFO] Import complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import DeepSeek BF16 weights into NeMo")
    parser.add_argument("--bf16-dir", required=True, help="Directory containing BF16 safetensors")
    parser.add_argument("--output", required=True, help="Path to write .nemo archive")
    parser.add_argument("--tensor-parallel", type=int, default=8, help="Tensor parallel world size")
    parser.add_argument("--pipeline-parallel", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument("--sequence-length", type=int, default=131072, help="Maximum context length")
    parser.add_argument("--model-name", default="deepseek_v3", help="NeMo model recipe identifier")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional JSON overrides (key=value JSON), e.g. parallel_attention=True",
    )
    args = parser.parse_args()

    overrides_dict = {}
    for override in args.override:
        key, value = override.split("=", maxsplit=1)
        overrides_dict[key] = json.loads(value)

    import_checkpoint(
        bf16_dir=Path(args.bf16_dir),
        output_path=Path(args.output),
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
        sequence_length=args.sequence_length,
        model_name=args.model_name,
        extra_overrides=overrides_dict,
    )


if __name__ == "__main__":
    main()

