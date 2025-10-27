import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file, safe_open

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    fp8_weight_names = []

    device = torch.device("cuda")
    device = "cuda"
    
    new_state_dict = {}
    for weight_name, safetensor_file in list(weight_map.items())[:256]:
        print(f"Processing {weight_name} from {safetensor_file}")
        with safe_open(os.path.join(fp8_path, safetensor_file), "pt", device="cpu") as f:
            weight = f.get_tensor(weight_name)

        if weight_name.endswith("_scale_inv"):
            continue
        elif weight.element_size() == 1:
            scale_inv_name = f"{weight_name}_scale_inv"
            fp8_weight_names.append(weight_name)
            safetensor_file_for_inv = os.path.join(fp8_path, weight_map[scale_inv_name])
            with safe_open(safetensor_file_for_inv, "pt", device="cpu") as f:
                scale_inv = f.get_tensor(scale_inv_name)
            new_state_dict[weight_name] = weight_dequant(weight.cuda(), scale_inv.cuda()).cpu()
        else:
            new_state_dict[weight_name] = weight
    # Split weights into 128 roughly equal parts
    NUM_FILES = 256
    weights_per_file = (len(new_state_dict) + NUM_FILES - 1) // NUM_FILES
    new_weight_map = {}
    
    for file_idx in range(NUM_FILES):
        start_idx = file_idx * weights_per_file
        end_idx = min((file_idx + 1) * weights_per_file, len(new_state_dict))
        
        file_weights = {}
        weight_names = list(new_state_dict.keys())[start_idx:end_idx]
        
        for weight_name in weight_names:
            file_weights[weight_name] = new_state_dict[weight_name]
            new_weight_map[weight_name] = f"model-{file_idx+1:05d}-of-00128.safetensors"
            
        new_safetensor_file = os.path.join(bf16_path, f"model-{file_idx+1:05d}-of-00128.safetensors")
        save_file(file_weights, new_safetensor_file)
    # Remove scale_inv entries from weight map
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        assert scale_inv_name in weight_map
        weight_map.pop(scale_inv_name)

    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)