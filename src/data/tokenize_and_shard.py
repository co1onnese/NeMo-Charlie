#!/usr/bin/env python3
"""
tokenize_and_shard.py
Tokenize HF dataset with special tokens for XML tags and actions.
Handles DeepSeek tokenizer specifics and creates shards for large datasets.

Usage:
  python src/data/tokenize_and_shard.py \
    --dataset_dir data/hf_datasets/sft_dataset \
    --tokenizer deepseek-ai/DeepSeek-V3.2-Exp \
    --out_dir data/hf_datasets/sft_dataset_tokenized
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from datasets import load_from_disk
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)

# Special tokens to add
XML_TAGS = [
    "<reasoning>", "</reasoning>",
    "<support>", "</support>",
    "<action>", "</action>"
]

ACTION_TOKENS = [
    "<STRONG_BUY>", "<BUY>", "<HOLD>", "<SELL>", "<STRONG_SELL>"
]

# Combine all special tokens
SPECIAL_TOKENS = XML_TAGS + ACTION_TOKENS


def load_and_extend_tokenizer(
    tokenizer_name: str,
    special_tokens: List[str],
    trust_remote_code: bool = True
) -> AutoTokenizer:
    """
    Load tokenizer and add special tokens.
    
    Args:
        tokenizer_name: Name or path of tokenizer
        special_tokens: List of special tokens to add
        trust_remote_code: Whether to trust remote code
        
    Returns:
        Extended tokenizer
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            use_fast=False  # DeepSeek may need slow tokenizer
        )
    except Exception as e:
        logger.warning(f"Failed to load with use_fast=False, trying use_fast=True: {e}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            use_fast=True
        )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # CRITICAL: Set padding side to 'right' for causal LM training
    tokenizer.padding_side = "right"
    logger.info(f"Set tokenizer padding_side to: {tokenizer.padding_side}")
    
    # Add special tokens
    num_added = tokenizer.add_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens to tokenizer")
    logger.info(f"Vocabulary size: {len(tokenizer)}")
    
    return tokenizer


def format_prompt(example: Dict[str, Any], template: str = "alpaca") -> Dict[str, Any]:
    """
    Format example into instruction-input-output template.
    
    Args:
        example: Dataset example
        template: Template name (alpaca, chatml, etc.)
        
    Returns:
        Dictionary with formatted fields
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output_text = example.get("output", "")
    
    if template == "alpaca":
        # Alpaca-style template
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full_text = prompt + output_text
        
    elif template == "chatml":
        # ChatML-style template (for DeepSeek)
        prompt = f"<|User|>: {instruction}\n{input_text}<|Assistant|>: "
        full_text = prompt + output_text
        
    else:
        # Simple template
        prompt = f"{instruction}\n{input_text}\n"
        full_text = prompt + output_text
    
    return {
        "prompt": prompt,
        "full_text": full_text,
        "output": output_text
    }


def tokenize_function(
    examples: Dict[str, List],
    tokenizer: AutoTokenizer,
    max_length: int = 2048,
    template: str = "alpaca"
) -> Dict[str, List]:
    """
    Tokenize examples with proper label masking.
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        template: Prompt template
        
    Returns:
        Dictionary with tokenized fields
    """
    # Format all examples
    formatted = [
        format_prompt(
            {
                "instruction": inst,
                "input": inp,
                "output": out
            },
            template=template
        )
        for inst, inp, out in zip(
            examples.get("instruction", [""]*len(examples["output"])),
            examples.get("input", [""]*len(examples["output"])),
            examples["output"]
        )
    ]
    
    # Tokenize full text
    full_texts = [f["full_text"] for f in formatted]
    prompts = [f["prompt"] for f in formatted]
    
    # Tokenize full sequences
    full_tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None
    )
    
    # Tokenize prompts to get their lengths
    prompt_tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # Create labels with -100 for prompt tokens and padding
    labels = []
    for i in range(len(full_tokenized["input_ids"])):
        input_ids = full_tokenized["input_ids"][i]
        attention_mask = full_tokenized["attention_mask"][i]
        prompt_length = len(prompt_tokenized["input_ids"][i])
        
        # Create labels: -100 for prompt and padding, actual ids for output
        label = []
        for j in range(len(input_ids)):
            if j < prompt_length:
                # Mask prompt tokens
                label.append(-100)
            elif attention_mask[j] == 0:
                # Mask padding tokens
                label.append(-100)
            else:
                # Keep actual response tokens
                label.append(input_ids[j])
        
        labels.append(label)
    
    return {
        "input_ids": full_tokenized["input_ids"],
        "attention_mask": full_tokenized["attention_mask"],
        "labels": labels
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize HF dataset with special tokens"
    )
    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Input HF dataset directory"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer name or path (default from .env)"
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for tokenized dataset (default: dataset_dir + '_tokenized')"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (default from .env)"
    )
    parser.add_argument(
        "--template",
        default="alpaca",
        choices=["alpaca", "chatml", "simple"],
        help="Prompt template"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code"
    )
    parser.add_argument(
        "--save_tokenizer",
        action="store_true",
        default=True,
        help="Save extended tokenizer"
    )
    
    args = parser.parse_args()
    
    # Get defaults from environment
    if args.tokenizer is None:
        args.tokenizer = os.getenv("BASE_MODEL", "deepseek-ai/DeepSeek-V3.2-Exp")
    
    if args.max_length is None:
        args.max_length = int(os.getenv("MAX_LENGTH", "2048"))
    
    if args.out_dir is None:
        args.out_dir = args.dataset_dir + "_tokenized"
    
    logger.info("=" * 80)
    logger.info("Tokenizing Dataset with Special Tokens")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset_dir}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    logger.info(f"Max length: {args.max_length}")
    logger.info(f"Template: {args.template}")
    logger.info(f"Output: {args.out_dir}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(args.dataset_dir)
    logger.info(f"Loaded dataset: {dataset}")
    
    # Load and extend tokenizer
    tokenizer = load_and_extend_tokenizer(
        args.tokenizer,
        SPECIAL_TOKENS,
        trust_remote_code=args.trust_remote_code
    )
    
    # Tokenize all splits
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(
            examples,
            tokenizer,
            max_length=args.max_length,
            template=args.template
        ),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )
    
    # Save tokenized dataset
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    tokenized_dataset.save_to_disk(args.out_dir)
    logger.info(f"Saved tokenized dataset to {args.out_dir}")
    
    # Save tokenizer
    if args.save_tokenizer:
        tokenizer_dir = os.path.join(args.out_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_dir)
        logger.info(f"Saved extended tokenizer to {tokenizer_dir}")
    
    # Print statistics
    logger.info("Tokenization statistics:")
    for split_name in tokenized_dataset.keys():
        split = tokenized_dataset[split_name]
        logger.info(f"  {split_name}: {len(split)} samples")
        
        # Sample first example
        if len(split) > 0:
            example = split[0]
            num_tokens = sum(1 for token in example["labels"] if token != -100)
            logger.info(f"    Example 0: {num_tokens} target tokens")
    
    logger.info("=" * 80)
    logger.info("Tokenization complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
