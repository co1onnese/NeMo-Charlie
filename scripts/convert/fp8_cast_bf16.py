"""DeepSeek FP8 to BF16 conversion script (patched).

This script is adapted from the official `deepseek-ai/DeepSeek-V3-Base`
distribution. It converts FP8-formatted weights into BF16 tensors that can be
ingested by NVIDIA NeMo. Compared to the upstream script, this version fixes
issues with partial weight iteration and ensures consistent metadata updates.

Usage:
    python scripts/convert/fp8_cast_bf16.py \
        --input-fp8-hf-path /path/to/deepseek-v3-fp8 \
        --output-bf16-hf-path /path/to/output-bf16
"""

import argparse
import json
import os
from glob import glob
from typing import Dict, Iterable

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from kernel import weight_dequant


def _load_index(fp8_path: str) -> Dict[str, str]:
    index_path = os.path.join(fp8_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with open(index_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["weight_map"]


def _iter_safetensor_files(fp8_path: str) -> Iterable[str]:
    files = sorted(glob(os.path.join(fp8_path, "*.safetensors")))
    for path in files:
        yield path


def convert(fp8_path: str, bf16_path: str) -> None:
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    weight_map = _load_index(fp8_path)
    fp8_weight_names = []

    # Cache to avoid repeated disk I/O; keep small to manage GPU memory
    loaded_files: Dict[str, Dict[str, torch.Tensor]] = {}

    def get_tensor(tensor_name: str) -> torch.Tensor:
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            # load_file returns tensors on the specified device
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(_iter_safetensor_files(fp8_path))
    for safetensor_file in tqdm(safetensor_files, desc="Converting FP8→BF16"):
        file_name = os.path.basename(safetensor_file)
        state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = state_dict

        new_state_dict: Dict[str, torch.Tensor] = {}
        for weight_name, weight in state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            if weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(
                        f"[WARN] Missing scale_inv tensor for {weight_name}; keeping FP8 weight"
                    )
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight

        save_file(new_state_dict, os.path.join(bf16_path, file_name))

        # retain only the two most recent files to limit GPU memory usage
        while len(loaded_files) > 2:
            oldest = next(iter(loaded_files))
            del loaded_files[oldest]
            torch.cuda.empty_cache()

    # Update model index to drop scale_inv entries referencing FP8 tensors
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        weight_map.pop(scale_inv_name, None)

    index_path = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump({"metadata": {}, "weight_map": weight_map}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepSeek FP8 weights to BF16")
    parser.add_argument("--input-fp8-hf-path", required=True, help="Path to FP8 checkpoint")
    parser.add_argument("--output-bf16-hf-path", required=True, help="Output path for BF16 weights")
    args = parser.parse_args()

    convert(args.input_fp8_hf_path, args.output_bf16_hf_path)


if __name__ == "__main__":
    main()
