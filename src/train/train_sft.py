#!/usr/bin/env python3
"""
train_sft.py
SFT training entrypoint using TRL.SFTTrainer + PEFT(LoRA) + QLoRA(bitsandbytes 4-bit).
Optimized for DeepSeek-V3.2-Exp with special token handling.

Usage:
  python src/train/train_sft.py --config configs/sft_config.yaml
  python src/train/train_sft.py --config configs/sft_config.yaml --smoke_test
"""
import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger
from src.utils.manifest import create_manifest

# Load environment
load_dotenv()

logger = setup_logger(__name__)

# Special tokens for XML tags and actions
SPECIAL_TOKENS = [
    "<reasoning>", "</reasoning>",
    "<support>", "</support>",
    "<action>", "</action>",
    "<STRONG_BUY>", "<BUY>", "<HOLD>", "<SELL>", "<STRONG_SELL>"
]


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge config with environment variables."""
    # Override with env vars if present
    env_overrides = {
        "base_model": "BASE_MODEL",
        "dataset_path": "HF_DATASET_DIR",
        "output_dir": "OUTPUT_DIR",
        "max_length": "MAX_LENGTH",
        "num_train_epochs": "NUM_TRAIN_EPOCHS",
        "learning_rate": "LEARNING_RATE",
        "per_device_train_batch_size": "PER_DEVICE_TRAIN_BATCH_SIZE",
        "gradient_accumulation_steps": "GRADIENT_ACCUMULATION_STEPS",
        "use_wandb": "USE_WANDB",
        "wandb_project": "WANDB_PROJECT",
    }
    
    for cfg_key, env_key in env_overrides.items():
        if env_key in os.environ:
            env_val = os.getenv(env_key)
            # Type conversion
            if cfg_key in ["max_length", "num_train_epochs", "per_device_train_batch_size", "gradient_accumulation_steps"]:
                config[cfg_key] = int(env_val)
            elif cfg_key in ["learning_rate"]:
                config[cfg_key] = float(env_val)
            elif cfg_key == "use_wandb":
                config[cfg_key] = env_val.lower() in ["true", "1", "yes"]
            else:
                config[cfg_key] = env_val
    
    return config


def setup_tokenizer(
    model_name: str,
    special_tokens: List[str],
    trust_remote_code: bool = True
) -> AutoTokenizer:
    """
    Load and configure tokenizer with special tokens.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=False  # DeepSeek may need slow tokenizer
        )
    except Exception as e:
        logger.warning(f"Failed with use_fast=False, trying use_fast=True: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_fast=True
        )
    
    # Set padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token = eos_token")
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            logger.info("Added [PAD] token")
    
    # Add special tokens
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)
    logger.info(f"Added {num_added} special tokens")
    logger.info(f"Vocabulary size: {len(tokenizer)}")
    
    return tokenizer


def format_example(example: Dict[str, Any]) -> Dict[str, str]:
    """Format example into prompt template."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    # Alpaca-style template
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    full_text = prompt + output_text
    
    return {
        "text": full_text,
        "prompt": prompt,
        "completion": output_text
    }


def load_model(
    model_name: str,
    config: Dict[str, Any],
    tokenizer_len: int
) -> torch.nn.Module:
    """
    Load model with quantization and PEFT configuration.
    """
    logger.info(f"Loading model: {model_name}")
    
    # Check if CPU only mode
    cpu_only = os.getenv("CPU_ONLY_MODE", "false").lower() == "true"
    
    if cpu_only:
        logger.warning("CPU_ONLY_MODE enabled - loading without quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=config.get("trust_remote_code", True),
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
    else:
        # BitsAndBytes config for QLoRA
        bnb_config = None
        if config.get("load_in_4bit", True):
            compute_dtype = torch.bfloat16 if config.get("bnb_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=config.get("bnb_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True)
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if config.get("bf16", False) else torch.float16,
            trust_remote_code=config.get("trust_remote_code", True),
            low_cpu_mem_usage=True,
            attn_implementation=config.get("attention_impl", None)
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )
    
    # Resize token embeddings for special tokens
    if tokenizer_len > model.config.vocab_size:
        logger.info(f"Resizing token embeddings from {model.config.vocab_size} to {tokenizer_len}")
        model.resize_token_embeddings(tokenizer_len)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=config.get("lora_dropout", 0.05),
        bias=config.get("bias", "none"),
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="SFT Training with DeepSeek")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run smoke test (10 steps)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("SFT Training - DeepSeek-V3.2-Exp")
    logger.info("=" * 80)
    
    # Load and merge config
    config = load_config(args.config)
    config = merge_config_with_env(config)
    
    # Smoke test overrides
    if args.smoke_test:
        logger.warning("SMOKE TEST MODE - limiting to 10 steps")
        config["num_train_epochs"] = 1
        config["save_steps"] = 5
        config["eval_steps"] = 5
        config["logging_steps"] = 1
        args.max_steps = 10
    
    logger.info(f"Model: {config['base_model']}")
    logger.info(f"Dataset: {config['dataset_path']}")
    logger.info(f"Output: {config['output_dir']}")
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        dataset = load_from_disk(config["dataset_path"])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error("Make sure to run convert_dataset.py first!")
        sys.exit(1)
    
    logger.info(f"Dataset splits: {list(dataset.keys())}")
    for split in dataset.keys():
        logger.info(f"  {split}: {len(dataset[split])} samples")
    
    # Setup tokenizer
    tokenizer = setup_tokenizer(
        config["base_model"],
        SPECIAL_TOKENS,
        trust_remote_code=config.get("trust_remote_code", True)
    )
    
    # Prepare dataset with formatting
    logger.info("Formatting dataset...")
    
    def formatting_func(example):
        return format_example(example)["text"]
    
    # Load model
    model = load_model(config["base_model"], config, len(tokenizer))
    
    # Training arguments
    output_dir = config.get("output_dir", "checkpoints/sft-run")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 2),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        learning_rate=config.get("learning_rate", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 200),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        logging_steps=config.get("logging_steps", 50),
        save_steps=config.get("save_steps", 500),
        eval_steps=config.get("eval_steps", 500),
        eval_strategy="steps" if "validation" in dataset else "no",
        save_total_limit=config.get("save_total_limit", 2),
        fp16=config.get("fp16", False) and torch.cuda.is_available(),
        bf16=config.get("bf16", True) and torch.cuda.is_available(),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        max_steps=args.max_steps if args.max_steps else -1,
        report_to="wandb" if config.get("use_wandb", False) else "none",
        run_name=f"sft-{config['base_model'].split('/')[-1]}",
        push_to_hub=False,
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        tokenizer=tokenizer,
        formatting_func=formatting_func,
        max_seq_length=config.get("max_length", 2048),
        packing=False,  # Don't pack sequences for our use case
    )
    
    # Save config and create manifest
    logger.info("Creating training manifest...")
    manifest = create_manifest(
        run_name=f"sft-{config['base_model'].split('/')[-1]}",
        config=config,
        output_path=os.path.join(output_dir, "manifest.json")
    )
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 80)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
