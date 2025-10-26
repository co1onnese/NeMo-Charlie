"""
validation.py
Data validation utilities for the SFT pipeline.
Validates XML structure, dates, actions, and prevents data leakage.
"""
import re
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from pathlib import Path


# Valid action values
VALID_ACTIONS = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}

# Date format regex
DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_date_format(date_str: str) -> bool:
    """
    Validate that date string is in ISO format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not date_str:
        return False
    if not DATE_REGEX.match(date_str):
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def normalize_date(date_str: str) -> Optional[str]:
    """
    Normalize various date formats to ISO format (YYYY-MM-DD).
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        ISO formatted date string or None if cannot parse
    """
    if not date_str:
        return None
    
    date_str = date_str.strip()
    
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y%m%d",
        "%d.%m.%Y"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue
    
    # Try to extract YYYY-MM-DD pattern
    match = re.search(r"(\d{4}-\d{2}-\d{2})", date_str)
    if match:
        return match.group(1)
    
    return None


def validate_action(action: str) -> bool:
    """
    Validate that action is one of the allowed values.
    
    Args:
        action: Action string to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not action:
        return False
    action_upper = action.strip().upper().replace(" ", "_")
    return action_upper in VALID_ACTIONS


def normalize_action(action: str) -> Optional[str]:
    """
    Normalize action to standard format.
    
    Args:
        action: Action string (case-insensitive, spaces/underscores flexible)
        
    Returns:
        Normalized action or None if invalid
    """
    if not action:
        return None
    
    action_upper = action.strip().upper().replace(" ", "_")
    
    # Direct match
    if action_upper in VALID_ACTIONS:
        return action_upper
    
    # Fuzzy matching
    if "STRONG" in action_upper and "BUY" in action_upper:
        return "STRONG_BUY"
    elif "STRONG" in action_upper and "SELL" in action_upper:
        return "STRONG_SELL"
    elif "BUY" in action_upper:
        return "BUY"
    elif "SELL" in action_upper:
        return "SELL"
    elif "HOLD" in action_upper:
        return "HOLD"
    
    return None


def validate_ticker(ticker: str) -> bool:
    """
    Basic validation of ticker symbol.
    
    Args:
        ticker: Ticker symbol
        
    Returns:
        True if valid format, False otherwise
    """
    if not ticker:
        return False
    ticker = ticker.strip()
    # Basic rules: 1-5 uppercase letters, possibly with dots/dashes
    if not re.match(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$", ticker):
        return False
    return True


def validate_thesis_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single thesis record.
    
    Args:
        record: Dictionary with thesis data
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields
    required_fields = ["ticker", "as_of_date", "reasoning", "action"]
    for field in required_fields:
        if field not in record or not record[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate ticker
    if "ticker" in record and record["ticker"]:
        if not validate_ticker(record["ticker"]):
            errors.append(f"Invalid ticker format: {record['ticker']}")
    
    # Validate date
    if "as_of_date" in record and record["as_of_date"]:
        if not validate_date_format(record["as_of_date"]):
            errors.append(f"Invalid date format: {record['as_of_date']} (expected YYYY-MM-DD)")
    
    # Validate action
    if "action" in record and record["action"]:
        if not validate_action(record["action"]):
            errors.append(f"Invalid action: {record['action']} (expected one of {VALID_ACTIONS})")
    
    # Check reasoning length
    if "reasoning" in record and record["reasoning"]:
        reasoning = record["reasoning"].strip()
        if len(reasoning) < 50:
            errors.append(f"Reasoning too short: {len(reasoning)} chars (minimum 50)")
        if len(reasoning) > 10000:
            errors.append(f"Reasoning too long: {len(reasoning)} chars (maximum 10000)")
    
    return len(errors) == 0, errors


def validate_time_splits(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    date_col: str = "as_of_date"
) -> Tuple[bool, List[str]]:
    """
    Validate that time-based splits don't have data leakage.
    
    Args:
        df: DataFrame with date column
        train_end: End date for training (YYYY-MM-DD)
        test_start: Start date for testing (YYYY-MM-DD)
        date_col: Name of date column
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Ensure date column exists
    if date_col not in df.columns:
        errors.append(f"Date column '{date_col}' not found in dataset")
        return False, errors
    
    # Convert to datetime
    try:
        dates = pd.to_datetime(df[date_col])
    except Exception as e:
        errors.append(f"Cannot parse dates in column '{date_col}': {e}")
        return False, errors
    
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    
    # Check for overlap
    if train_end_dt >= test_start_dt:
        errors.append(f"Train end date ({train_end}) must be before test start date ({test_start})")
    
    # Check that there's a gap (optional but recommended)
    gap_days = (test_start_dt - train_end_dt).days
    if gap_days < 1:
        errors.append(f"No gap between train and test sets (recommended at least 1 day)")
    
    # Verify splits
    train_mask = dates <= train_end_dt
    test_mask = dates >= test_start_dt
    
    train_count = train_mask.sum()
    test_count = test_mask.sum()
    
    if train_count == 0:
        errors.append("No training samples found before train_end date")
    if test_count == 0:
        errors.append("No test samples found after test_start date")
    
    # Check for any dates in the gap (validation set)
    gap_mask = (dates > train_end_dt) & (dates < test_start_dt)
    gap_count = gap_mask.sum()
    
    return len(errors) == 0, errors


def validate_dataset_statistics(
    df: pd.DataFrame,
    min_samples: int = 10,
    max_samples: Optional[int] = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate dataset statistics and generate report.
    
    Args:
        df: DataFrame to validate
        min_samples: Minimum required samples
        max_samples: Maximum allowed samples (optional)
        
    Returns:
        Tuple of (is_valid, list_of_warnings, statistics_dict)
    """
    warnings = []
    stats = {}
    
    # Basic counts
    stats["total_samples"] = len(df)
    stats["unique_tickers"] = df["ticker"].nunique() if "ticker" in df.columns else 0
    
    if len(df) < min_samples:
        warnings.append(f"Dataset has only {len(df)} samples (minimum {min_samples})")
    
    if max_samples and len(df) > max_samples:
        warnings.append(f"Dataset has {len(df)} samples (maximum {max_samples})")
    
    # Check for duplicates
    if "uid" in df.columns:
        duplicates = df["uid"].duplicated().sum()
        stats["duplicates"] = duplicates
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate UIDs")
    
    # Action distribution
    if "action" in df.columns:
        action_dist = df["action"].value_counts().to_dict()
        stats["action_distribution"] = action_dist
        
        # Check for class imbalance
        if len(action_dist) > 0:
            max_count = max(action_dist.values())
            min_count = min(action_dist.values())
            if max_count > 10 * min_count:
                warnings.append(f"Severe class imbalance detected (ratio {max_count/min_count:.1f}:1)")
    
    # Date range
    if "as_of_date" in df.columns:
        try:
            dates = pd.to_datetime(df["as_of_date"])
            stats["date_range"] = {
                "min": dates.min().isoformat(),
                "max": dates.max().isoformat(),
                "span_days": (dates.max() - dates.min()).days
            }
        except Exception:
            warnings.append("Cannot compute date range statistics")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        stats["missing_values"] = missing[missing > 0].to_dict()
        warnings.append(f"Found missing values in {len(missing[missing > 0])} columns")
    
    return len(warnings) == 0, warnings, stats


def check_xml_structure(xml_content: str) -> Tuple[bool, List[str]]:
    """
    Basic validation of XML structure without full parsing.
    
    Args:
        xml_content: XML content as string
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for basic XML structure
    if not xml_content.strip().startswith("<?xml") and not xml_content.strip().startswith("<"):
        errors.append("Content does not appear to be XML")
    
    # Check for required tags
    required_tags = ["thesis", "reasoning", "action"]
    for tag in required_tags:
        if f"<{tag}" not in xml_content and f"<{tag}>" not in xml_content:
            errors.append(f"Missing required tag: <{tag}>")
    
    # Check for balanced tags (simple check)
    for tag in required_tags:
        open_count = xml_content.count(f"<{tag}>") + xml_content.count(f"<{tag} ")
        close_count = xml_content.count(f"</{tag}>")
        if open_count != close_count:
            errors.append(f"Unbalanced <{tag}> tags (open: {open_count}, close: {close_count})")
    
    return len(errors) == 0, errors
