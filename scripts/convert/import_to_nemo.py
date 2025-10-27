#!/usr/bin/env python3
"""Import converted DeepSeek-V3 BF16 weights into a NeMo .nemo archive.

This script assumes the FP8→BF16 conversion has already been performed using
`scripts/convert/fp8_cast_bf16.py`. It uses the NeMo LLM import API to convert
HuggingFace format checkpoints to NeMo's .nemo archive format.

Example:
    python scripts/convert/import_to_nemo.py \
        --bf16-dir /data/models/deepseek-v3-bf16 \
        --output /data/models/deepseek-v3-base_tp8_pp1.nemo

The script performs minimal validation of the input directory layout but does
not attempt to download or convert weights itself. Install NeMo and its
dependencies prior to running (see `requirements_nemo.txt`).

Official NeMo docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html
"""

import argparse
import os
import sys
from pathlib import Path

# CRITICAL FIX: Monkey patch TransformerConfig before NeMo imports it
# This forces persist_layer_norm=False and gradient_accumulation_fusion=False at the Megatron level
print("[INFO] Applying Megatron-Core compatibility patches...")
import megatron.core.transformer.transformer_config as tf_config_module
original_transformer_config_init = tf_config_module.TransformerConfig.__init__

def patched_transformer_config_init(self, *args, **kwargs):
    """Force persist_layer_norm=False for PyTorch LayerNorm compatibility
    and gradient_accumulation_fusion=False to avoid APEX CUDA extension requirement"""
    kwargs['persist_layer_norm'] = False
    kwargs['gradient_accumulation_fusion'] = False
    return original_transformer_config_init(self, *args, **kwargs)

tf_config_module.TransformerConfig.__init__ = patched_transformer_config_init
print("[INFO] ✓ Forced persist_layer_norm=False and gradient_accumulation_fusion=False in TransformerConfig")

# AGGRESSIVE FIX: Patch the DeepSeekConfig dataclass field default BEFORE NeMo loads
# This is the nuclear option - directly modify the dataclass field
import megatron.core.model_parallel_config as mp_config_module
if hasattr(mp_config_module, 'ModelParallelConfig'):
    mpc_class = mp_config_module.ModelParallelConfig
    if hasattr(mpc_class, '__dataclass_fields__') and 'gradient_accumulation_fusion' in mpc_class.__dataclass_fields__:
        # Directly modify the field default
        field = mpc_class.__dataclass_fields__['gradient_accumulation_fusion']
        field.default = False
        print("[INFO] ✓ Modified ModelParallelConfig.gradient_accumulation_fusion default to False")

# Also patch the __post_init__ to force it
original_mpc_post_init = getattr(mp_config_module.ModelParallelConfig, '__post_init__', None)
if original_mpc_post_init:
    def patched_mpc_post_init(self):
        """Force gradient_accumulation_fusion=False before calling original __post_init__"""
        object.__setattr__(self, 'gradient_accumulation_fusion', False)
        if original_mpc_post_init:
            return original_mpc_post_init(self)
    mp_config_module.ModelParallelConfig.__post_init__ = patched_mpc_post_init
    print("[INFO] ✓ Patched ModelParallelConfig.__post_init__ to force gradient_accumulation_fusion=False")

# Try to import NeMo LLM, handle modelopt issues
try:
    from nemo.collections import llm
    # Verify import_ckpt is available
    if not hasattr(llm, 'import_ckpt'):
        raise ImportError("llm.import_ckpt not available - modelopt dependency issue")
except Exception as e:
    print(f"[ERROR] Failed to import NeMo LLM properly: {e}")
    print("[ERROR]")
    print("[ERROR] This is likely due to a broken modelopt installation.")
    print("[ERROR] Please fix your environment:")
    print("[ERROR]")
    print("[ERROR]   cd /workspace/NeMo-Charlie")
    print("[ERROR]   source venv/bin/activate")
    print("[ERROR]   pip uninstall -y modelopt nvidia-modelopt")
    print("[ERROR]   pip install nvidia-modelopt")
    print("[ERROR]   # Or if that fails:")
    print("[ERROR]   pip uninstall -y modelopt")
    print("[ERROR]")
    sys.exit(1)


def import_checkpoint(
    bf16_dir: Path,
    output_path: Path,
    tensor_parallel: int = 8,
    pipeline_parallel: int = 1,
) -> None:
    """Import DeepSeek V3 BF16 checkpoint to NeMo format.

    Args:
        bf16_dir: Path to HuggingFace BF16 checkpoint directory
        output_path: Path where .nemo archive will be written
        tensor_parallel: Tensor model parallel size (default: 8 for 8xH100)
        pipeline_parallel: Pipeline model parallel size (default: 1)
    """
    if not bf16_dir.exists():
        raise FileNotFoundError(f"BF16 directory not found: {bf16_dir}")

    # Verify required files exist
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not (bf16_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {bf16_dir / file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] ========================================")
    print("[INFO] NeMo DeepSeek-V3 Import with Tensor Parallelism")
    print("[INFO] ========================================")
    print(f"[INFO] BF16 source: {bf16_dir}")
    print(f"[INFO] Output archive: {output_path}")
    print(f"[INFO] Tensor Parallel Size: {tensor_parallel}")
    print(f"[INFO] Pipeline Parallel Size: {pipeline_parallel}")
    print("[INFO] gradient_accumulation_fusion: False (no APEX required)")
    print("[INFO] ========================================")
    print("[INFO] Model size: 671B parameters (~1.3TB)")
    print("[INFO] This will take 10-30 minutes to load and convert...")
    print("[INFO] ========================================")

    # CRITICAL: Use tensor parallelism to split 671B model across GPUs
    # Without TP=8, the 1.3TB model cannot fit in memory
    # Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html
    config = llm.DeepSeekV3Config(
        gradient_accumulation_fusion=False,
        tensor_model_parallel_size=tensor_parallel,
        pipeline_model_parallel_size=pipeline_parallel,
    )
    
    print(f"[INFO] Creating DeepSeekModel with TP={tensor_parallel}, PP={pipeline_parallel}")
    model = llm.DeepSeekModel(config)
    
    print("[INFO] Starting checkpoint import (this is the slow part)...")
    print("[INFO] Progress: Loading 163 safetensors files (~1.3TB)...")
    
    llm.import_ckpt(
        model=model,
        source=f"hf://{bf16_dir.absolute()}",
        output_path=str(output_path),
    )

    print("[INFO] ========================================")
    print("[INFO] ✓ Import complete!")
    print(f"[INFO] ✓ NeMo checkpoint saved to: {output_path}")
    print("[INFO] ========================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import DeepSeek V3 BF16 weights into NeMo format",
        epilog="Example: python import_to_nemo.py --bf16-dir /data/models/deepseek-v3-bf16 --output /data/models/deepseek-v3-base.nemo"
    )
    parser.add_argument(
        "--bf16-dir",
        required=True,
        help="Directory containing BF16 HuggingFace checkpoint (config.json, safetensors, etc.)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path where .nemo archive will be written"
    )
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=8,
        help="Tensor model parallel size (default: 8 for 8×H100). REQUIRED for 671B model."
    )
    parser.add_argument(
        "--pipeline-parallel",
        type=int,
        default=1,
        help="Pipeline model parallel size (default: 1)"
    )
    # Legacy arguments kept for backwards compatibility but ignored
    parser.add_argument("--sequence-length", type=int, default=131072, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default="deepseek_v3", help=argparse.SUPPRESS)
    parser.add_argument("--override", action="append", default=[], help=argparse.SUPPRESS)

    args = parser.parse_args()

    print(f"[INFO] Using Tensor Parallel={args.tensor_parallel}, Pipeline Parallel={args.pipeline_parallel}")
    print("[INFO] For 671B DeepSeek-V3: TP=8 is recommended for 8×H100 (splits model across GPUs)")
    print("[INFO] This script must be run with torchrun for multi-GPU:")
    print("[INFO]   torchrun --nproc_per_node=8 scripts/convert/import_to_nemo.py ...")
    print()

    import_checkpoint(
        bf16_dir=Path(args.bf16_dir),
        output_path=Path(args.output),
        tensor_parallel=args.tensor_parallel,
        pipeline_parallel=args.pipeline_parallel,
    )


if __name__ == "__main__":
    main()

