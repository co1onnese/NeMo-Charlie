#!/usr/bin/env python3
"""
train_sft.py
SFT training entrypoint using TRL.SFTTrainer + PEFT(LoRA) + QLoRA(bitsandbytes 4-bit).
Reads a YAML config or command-line overrides.

Usage:
  python src/train/train_sft.py --config configs/sft_config.yaml
"""
import argparse
import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def add_special_tokens(tokenizer, tokens):
    added = tokenizer.add_tokens(tokens)
    if added:
        logger.info("Added %d special tokens", added)
    return added

def build_prompt_fields(example):
    """
    Build {full, input_prefix, target} fields expected by the tokenizer mapping later.
    Example dataset expected to have 'instruction', 'input', 'output'
    """
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")
    input_prefix = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    full = input_prefix + out
    return {"full": full, "input_prefix": input_prefix, "target": out}

def tokenize_and_create_labels(batch, tokenizer, max_length):
    """
    Tokenize 'full' and 'input_prefix' and create 'labels' that are -100 for prefix
    and actual token ids for target portion.
    Operates on batched inputs from HF dataset map.
    """
    fulls = batch["full"]
    prefixes = batch["input_prefix"]
    targets = batch["target"]
    all_input_ids = []
    all_attn = []
    all_labels = []
    for full, prefix, target in zip(fulls, prefixes, targets):
        tok_full = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
        tok_prefix = tokenizer(prefix, truncation=True, max_length=max_length, padding="max_length")
        # tokenize target separately (no padding)
        tok_target = tokenizer(target, truncation=True, max_length=max_length, padding=False)
        input_ids = tok_full["input_ids"]
        attn = tok_full["attention_mask"]
        # create labels initialized to -100
        labels = [-100] * len(input_ids)
        t_ids = tok_target["input_ids"]
        # align target to the end of the full sequence (simple heuristic)
        if len(t_ids) <= len(input_ids):
            start = len(input_ids) - len(t_ids)
            labels[start:start + len(t_ids)] = t_ids
        else:
            # target bigger than max_length -> align last tokens
            labels = t_ids[-len(input_ids):]
        all_input_ids.append(input_ids)
        all_attn.append(attn)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "attention_mask": all_attn, "labels": all_labels}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config for training")
    args = p.parse_args()
    cfg = load_yaml(args.config)

    # Load dataset (arrow dataset saved by convert_dataset.py)
    ds = load_from_disk(cfg["dataset_path"])
    if "train" in ds:
        ds_train = ds["train"]
        ds_eval = ds.get("validation", None)
    else:
        ds = ds.train_test_split(test_size=cfg.get("validation_fraction", 0.05), seed=cfg.get("seed", 42))
        ds_train = ds["train"]; ds_eval = ds["test"]

    # Build prompt fields
    ds_train = ds_train.map(lambda ex: build_prompt_fields(ex), remove_columns=ds_train.column_names)
    if ds_eval is not None:
        ds_eval = ds_eval.map(lambda ex: build_prompt_fields(ex), remove_columns=ds_eval.column_names)

    # Tokenizer & model init
    tokenizer = AutoTokenizer.from_pretrained(cfg.get("tokenizer", cfg["base_model"]), use_fast=False)
    special_tokens = cfg.get("special_tokens", [])
    added = add_special_tokens(tokenizer, special_tokens)

    # Map tokenization
    ds_train = ds_train.map(lambda batch: tokenize_and_create_labels(batch, tokenizer, cfg["max_length"]),
                            batched=True, remove_columns=ds_train.column_names)
    if ds_eval is not None:
        ds_eval = ds_eval.map(lambda batch: tokenize_and_create_labels(batch, tokenizer, cfg["max_length"]),
                              batched=True, remove_columns=ds_eval.column_names)

    # data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, padding="max_length")

    # BitsAndBytes (QLoRA) config
    bnb_config = None
    if cfg.get("load_in_4bit", True):
        compute_dtype = torch.bfloat16 if cfg.get("bnb_compute_dtype", "bfloat16") == "bfloat16" else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get("bnb_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype
        )

    logger.info("Loading model (this can take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config if bnb_config else None,
        device_map="auto",
        torch_dtype=torch.float16 if cfg.get("fp16", True) else None,
        low_cpu_mem_usage=True,
        trust_remote_code=cfg.get("trust_remote_code", False)
    )

    if added:
        model.resize_token_embeddings(len(tokenizer))

    # Prepare for kbit and apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    logger.info("PEFT model prepared; trainable params should be small.")

    # TRL SFTConfig
    sft_cfg = SFTConfig(
        model_name_or_path=cfg["base_model"],
        learning_rate=cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        logging_steps=cfg.get("logging_steps", 100),
        save_steps=cfg.get("save_steps", 2000),
        eval_steps=cfg.get("eval_steps", 2000),
        output_dir=cfg.get("output_dir", "checkpoints/sft-run"),
        save_total_limit=cfg.get("save_total_limit", 3),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),
        warmup_steps=cfg.get("warmup_steps", 100),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_cfg,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator
    )

    # optional wandb
    if cfg.get("use_wandb"):
        import wandb
        wandb.init(project=cfg.get("wandb_project", "sft-trading"))

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished. Saving model and tokenizer...")
    trainer.save_model(cfg.get("output_dir", "checkpoints/sft-run"))
    tokenizer.save_pretrained(cfg.get("output_dir", "checkpoints/sft-run"))

if __name__ == "__main__":
    main()
