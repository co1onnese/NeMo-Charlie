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

# ADDITIONAL FIX: Patch ColumnParallelLinear to disable gradient_accumulation_fusion
import megatron.core.tensor_parallel.layers as tp_layers
original_column_parallel_init = tp_layers.ColumnParallelLinear.__init__

def patched_column_parallel_init(self, *args, **kwargs):
    """Force gradient_accumulation_fusion=False to avoid APEX CUDA extension requirement
    
    ColumnParallelLinear reads gradient_accumulation_fusion from the config object,
    so we need to patch the config if it's passed.
    """
    # If config is passed as kwarg or positional arg, patch it
    config = kwargs.get('config', None)
    if config is None and len(args) > 5:  # config is typically a later positional arg
        config = args[5] if len(args) > 5 else None
    
    if config and hasattr(config, 'gradient_accumulation_fusion'):
        # Monkey patch the config object's attribute
        object.__setattr__(config, 'gradient_accumulation_fusion', False)
    
    if 'gradient_accumulation_fusion' in kwargs:
        kwargs['gradient_accumulation_fusion'] = False
    
    return original_column_parallel_init(self, *args, **kwargs)

tp_layers.ColumnParallelLinear.__init__ = patched_column_parallel_init
print("[INFO] ✓ Patched ColumnParallelLinear to disable gradient_accumulation_fusion")

# Also patch RowParallelLinear for consistency
original_row_parallel_init = tp_layers.RowParallelLinear.__init__

def patched_row_parallel_init(self, *args, **kwargs):
    """Force gradient_accumulation_fusion=False to avoid APEX CUDA extension requirement"""
    config = kwargs.get('config', None)
    if config is None and len(args) > 5:
        config = args[5] if len(args) > 5 else None
    
    if config and hasattr(config, 'gradient_accumulation_fusion'):
        object.__setattr__(config, 'gradient_accumulation_fusion', False)
    
    if 'gradient_accumulation_fusion' in kwargs:
        kwargs['gradient_accumulation_fusion'] = False
    
    return original_row_parallel_init(self, *args, **kwargs)

tp_layers.RowParallelLinear.__init__ = patched_row_parallel_init
print("[INFO] ✓ Patched RowParallelLinear to disable gradient_accumulation_fusion")

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
) -> None:
    """Import DeepSeek V3 BF16 checkpoint to NeMo format.

    Args:
        bf16_dir: Path to HuggingFace BF16 checkpoint directory
        output_path: Path where .nemo archive will be written
    """
    if not bf16_dir.exists():
        raise FileNotFoundError(f"BF16 directory not found: {bf16_dir}")

    # Verify required files exist
    required_files = ["config.json", "tokenizer.json"]
    for file in required_files:
        if not (bf16_dir / file).exists():
            raise FileNotFoundError(f"Required file not found: {bf16_dir / file}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Importing BF16 checkpoint into NeMo archive…")
    print(f"[INFO] BF16 source: {bf16_dir}")
    print(f"[INFO] Output archive: {output_path}")
    print("[INFO] Using DeepSeekV3Config with gradient_accumulation_fusion=False")
    print("[INFO] This may take 5-15 minutes...")

    # Disable gradient_accumulation_fusion since APEX with CUDA extensions is not available
    # Reference: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html
    llm.import_ckpt(
        model=llm.DeepSeekModel(llm.DeepSeekV3Config(gradient_accumulation_fusion=False)),
        source=f"hf://{bf16_dir.absolute()}",
        output_path=str(output_path),
    )

    print("[INFO] Import complete.")
    print(f"[INFO] NeMo checkpoint saved to: {output_path}")


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
    # Legacy arguments kept for backwards compatibility but ignored
    parser.add_argument("--tensor-parallel", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--pipeline-parallel", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, default=131072, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default="deepseek_v3", help=argparse.SUPPRESS)
    parser.add_argument("--override", action="append", default=[], help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Note: Tensor/pipeline parallel settings are now handled by NeMo's DeepSeekV3Config
    # and the training recipe, not at import time
    if args.tensor_parallel != 8 or args.pipeline_parallel != 1:
        print("[WARNING] --tensor-parallel and --pipeline-parallel are deprecated.")
        print("[WARNING] Parallelism is configured in the NeMo training recipe, not at import.")

    import_checkpoint(
        bf16_dir=Path(args.bf16_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

