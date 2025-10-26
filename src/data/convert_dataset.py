#!/usr/bin/env python3
"""
convert_dataset.py
Convert JSONL to HuggingFace Dataset with strict time-based train/validation/test splits.
Validates data integrity and prevents time leakage.

Usage:
  python src/data/convert_dataset.py --jsonl data/jsonl/all.jsonl --out_dir data/hf_datasets/sft_dataset
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.validation import (
    validate_thesis_record,
    validate_time_splits,
    validate_dataset_statistics,
    normalize_date,
    normalize_action
)
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

logger = setup_logger(__name__)


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Load JSONL file and return list of records.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
    
    logger.info(f"Loaded {len(records)} records from {file_path}")
    return records


def normalize_records(records: List[Dict]) -> List[Dict]:
    """
    Normalize and validate records.
    
    Args:
        records: List of raw records
        
    Returns:
        List of normalized records
    """
    normalized = []
    skipped = 0
    
    for i, record in enumerate(records):
        # Normalize date
        if "as_of_date" in record:
            normalized_date = normalize_date(record["as_of_date"])
            if normalized_date:
                record["as_of_date"] = normalized_date
            else:
                logger.warning(f"Record {i}: Invalid date format, skipping")
                skipped += 1
                continue
        
        # Normalize action
        if "action" in record:
            normalized_action = normalize_action(record["action"])
            if normalized_action:
                record["action"] = normalized_action
            else:
                logger.warning(f"Record {i}: Invalid action, skipping")
                skipped += 1
                continue
        
        # Validate record
        is_valid, errors = validate_thesis_record(record)
        if not is_valid:
            logger.warning(f"Record {i} validation failed: {errors}")
            skipped += 1
            continue
        
        normalized.append(record)
    
    logger.info(f"Normalized {len(normalized)} records, skipped {skipped}")
    return normalized


def create_time_splits(
    df: pd.DataFrame,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    validation_days: int = 30,
    date_col: str = "as_of_date"
) -> Dict[str, pd.DataFrame]:
    """
    Create time-based train/validation/test splits.
    
    Args:
        df: DataFrame with date column
        train_end: End date for training (from env or arg)
        test_start: Start date for testing (from env or arg)
        validation_days: Days to carve from end of training for validation
        date_col: Name of date column
        
    Returns:
        Dictionary with train/validation/test DataFrames
    """
    # Get dates from environment if not provided
    if train_end is None:
        train_end = os.getenv("TRAIN_END_DATE", "2024-12-31")
    if test_start is None:
        test_start = os.getenv("TEST_START_DATE", "2025-01-01")
    
    logger.info(f"Creating time splits: train_end={train_end}, test_start={test_start}")
    
    # Convert dates
    df[date_col] = pd.to_datetime(df[date_col])
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    
    # Validate splits
    is_valid, errors = validate_time_splits(df, train_end, test_start, date_col)
    if not is_valid:
        logger.error(f"Time split validation failed: {errors}")
        raise ValueError(f"Invalid time splits: {errors}")
    
    # Create validation split from end of training
    validation_start_dt = train_end_dt - timedelta(days=validation_days)
    
    # Split data
    train_mask = df[date_col] < validation_start_dt
    val_mask = (df[date_col] >= validation_start_dt) & (df[date_col] <= train_end_dt)
    test_mask = df[date_col] >= test_start_dt
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Convert dates back to strings for HF dataset
    for split_df in [train_df, val_df, test_df]:
        split_df[date_col] = split_df[date_col].dt.strftime("%Y-%m-%d")
    
    logger.info(f"Train samples: {len(train_df)} (up to {validation_start_dt.date()})")
    logger.info(f"Validation samples: {len(val_df)} ({validation_start_dt.date()} to {train_end})")
    logger.info(f"Test samples: {len(test_df)} (from {test_start})")
    
    # Verify no overlap
    train_dates = set(train_df[date_col])
    val_dates = set(val_df[date_col])
    test_dates = set(test_df[date_col])
    
    if train_dates & test_dates:
        logger.error("CRITICAL: Train and test sets overlap!")
        raise ValueError("Data leakage detected: train/test overlap")
    
    if val_dates & test_dates:
        logger.warning("Validation and test sets overlap (this may be intentional)")
    
    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df
    }


def create_hf_dataset(splits: Dict[str, pd.DataFrame]) -> DatasetDict:
    """
    Convert pandas DataFrames to HuggingFace DatasetDict.
    
    Args:
        splits: Dictionary with train/validation/test DataFrames
        
    Returns:
        HuggingFace DatasetDict
    """
    dataset_dict = {}
    
    for split_name, df in splits.items():
        if len(df) > 0:
            # Convert to dict format
            data = df.to_dict(orient="list")
            dataset_dict[split_name] = Dataset.from_dict(data)
            logger.info(f"Created {split_name} dataset with {len(df)} samples")
        else:
            logger.warning(f"Split '{split_name}' is empty, skipping")
    
    return DatasetDict(dataset_dict)


def save_dataset_metadata(
    output_dir: str,
    splits: Dict[str, pd.DataFrame],
    stats: Dict
):
    """
    Save dataset metadata and statistics.
    
    Args:
        output_dir: Output directory
        splits: Dictionary with splits
        stats: Statistics dictionary
    """
    metadata = {
        "created_at": datetime.now().isoformat(),
        "splits": {
            name: {
                "num_samples": len(df),
                "date_range": {
                    "min": df["as_of_date"].min() if len(df) > 0 else None,
                    "max": df["as_of_date"].max() if len(df) > 0 else None
                }
            }
            for name, df in splits.items()
        },
        "statistics": stats
    }
    
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to HuggingFace Dataset with time-based splits"
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for HF dataset"
    )
    parser.add_argument(
        "--train_end",
        default=None,
        help="End date for training (YYYY-MM-DD), default from .env"
    )
    parser.add_argument(
        "--test_start",
        default=None,
        help="Start date for testing (YYYY-MM-DD), default from .env"
    )
    parser.add_argument(
        "--validation_days",
        type=int,
        default=None,
        help="Number of days for validation split (default from .env or 30)"
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=10,
        help="Minimum samples per split"
    )
    
    args = parser.parse_args()
    
    # Get validation_days from env if not provided
    if args.validation_days is None:
        args.validation_days = int(os.getenv("VALIDATION_DAYS", "30"))
    
    logger.info("=" * 80)
    logger.info("Converting JSONL to HuggingFace Dataset")
    logger.info("=" * 80)
    
    # Load JSONL
    records = load_jsonl(args.jsonl)
    if len(records) == 0:
        logger.error("No records loaded from JSONL file")
        sys.exit(1)
    
    # Normalize and validate
    records = normalize_records(records)
    if len(records) == 0:
        logger.error("No valid records after normalization")
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Validate dataset statistics
    is_valid, warnings, stats = validate_dataset_statistics(df, min_samples=args.min_samples)
    if warnings:
        for warning in warnings:
            logger.warning(warning)
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Unique tickers: {stats['unique_tickers']}")
    if "action_distribution" in stats:
        logger.info(f"  Action distribution:")
        for action, count in stats["action_distribution"].items():
            logger.info(f"    {action}: {count}")
    
    # Create time-based splits
    splits = create_time_splits(
        df,
        train_end=args.train_end,
        test_start=args.test_start,
        validation_days=args.validation_days
    )
    
    # Verify minimum samples
    for split_name, split_df in splits.items():
        if len(split_df) < args.min_samples:
            logger.error(
                f"Split '{split_name}' has only {len(split_df)} samples "
                f"(minimum {args.min_samples})"
            )
            sys.exit(1)
    
    # Create HuggingFace dataset
    dataset_dict = create_hf_dataset(splits)
    
    # Save dataset
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(args.out_dir)
    logger.info(f"Saved dataset to {args.out_dir}")
    
    # Save metadata
    save_dataset_metadata(args.out_dir, splits, stats)
    
    logger.info("=" * 80)
    logger.info("Dataset conversion complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
