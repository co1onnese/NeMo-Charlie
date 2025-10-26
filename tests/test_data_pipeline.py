#!/usr/bin/env python3
"""
test_data_pipeline.py
Smoke test for data pipeline components.
Tests XML parsing, dataset conversion, and validation.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.logger import setup_logger
from src.utils.validation import (
    validate_date_format,
    validate_action,
    normalize_date,
    normalize_action,
    validate_thesis_record
)

logger = setup_logger(__name__)


def test_date_validation():
    """Test date validation and normalization."""
    logger.info("Testing date validation...")
    
    # Valid dates
    assert validate_date_format("2023-10-24"), "Valid ISO date failed"
    assert validate_date_format("2024-12-31"), "Valid ISO date failed"
    
    # Invalid dates
    assert not validate_date_format("2023/10/24"), "Should reject slashes"
    assert not validate_date_format("24-10-2023"), "Should reject EU format"
    assert not validate_date_format("invalid"), "Should reject invalid"
    
    # Normalization
    assert normalize_date("2023-10-24") == "2023-10-24", "ISO format should pass through"
    assert normalize_date("2023/10/24") == "2023-10-24", "Should normalize slashes"
    assert normalize_date("10/24/2023") == "2023-10-24", "Should normalize US format"
    
    logger.info("✓ Date validation tests passed")


def test_action_validation():
    """Test action validation and normalization."""
    logger.info("Testing action validation...")
    
    # Valid actions
    assert validate_action("BUY"), "BUY should be valid"
    assert validate_action("SELL"), "SELL should be valid"
    assert validate_action("HOLD"), "HOLD should be valid"
    assert validate_action("STRONG_BUY"), "STRONG_BUY should be valid"
    assert validate_action("STRONG_SELL"), "STRONG_SELL should be valid"
    
    # Case insensitive
    assert validate_action("buy"), "buy (lowercase) should be valid"
    assert validate_action("Hold"), "Hold (mixed) should be valid"
    
    # Invalid
    assert not validate_action("INVALID"), "Invalid action should fail"
    assert not validate_action(""), "Empty should fail"
    
    # Normalization
    assert normalize_action("buy") == "BUY", "Should normalize to uppercase"
    assert normalize_action("strong buy") == "STRONG_BUY", "Should handle spaces"
    assert normalize_action("STRONG_BUY") == "STRONG_BUY", "Should pass through"
    assert normalize_action("hold") == "HOLD", "Should normalize hold"
    
    logger.info("✓ Action validation tests passed")


def test_record_validation():
    """Test thesis record validation."""
    logger.info("Testing record validation...")
    
    # Valid record
    valid_record = {
        "ticker": "TSLA",
        "as_of_date": "2023-10-24",
        "reasoning": "This is a detailed reasoning with more than 50 characters to pass validation.",
        "action": "BUY"
    }
    
    is_valid, errors = validate_thesis_record(valid_record)
    assert is_valid, f"Valid record failed: {errors}"
    
    # Missing ticker
    invalid_record = valid_record.copy()
    invalid_record["ticker"] = None
    is_valid, errors = validate_thesis_record(invalid_record)
    assert not is_valid, "Should fail with missing ticker"
    
    # Invalid action
    invalid_record = valid_record.copy()
    invalid_record["action"] = "INVALID"
    is_valid, errors = validate_thesis_record(invalid_record)
    assert not is_valid, "Should fail with invalid action"
    
    # Reasoning too short
    invalid_record = valid_record.copy()
    invalid_record["reasoning"] = "Too short"
    is_valid, errors = validate_thesis_record(invalid_record)
    assert not is_valid, "Should fail with short reasoning"
    
    logger.info("✓ Record validation tests passed")


def test_xml_parsing():
    """Test XML parsing with sample data."""
    logger.info("Testing XML parsing...")
    
    from src.parsers.xml_to_jsonl import parse_file
    
    # Use the example file
    example_file = "data/samples/example_input.xml"
    if os.path.exists(example_file):
        records = parse_file(example_file)
        assert len(records) > 0, "Should parse at least one record"
        
        # Check first record
        if len(records) > 0:
            rec = records[0]
            assert "ticker" in rec, "Should have ticker"
            assert "reasoning" in rec, "Should have reasoning"
            assert "action" in rec, "Should have action"
            
            logger.info(f"  Parsed {len(records)} records from example file")
            logger.info(f"  First record ticker: {rec.get('ticker')}")
            logger.info(f"  First record action: {rec.get('action')}")
        
        logger.info("✓ XML parsing tests passed")
    else:
        logger.warning(f"Example file not found: {example_file}, skipping XML parsing test")


def main():
    """Run all data pipeline tests."""
    logger.info("=" * 80)
    logger.info("Data Pipeline Smoke Tests")
    logger.info("=" * 80)
    
    try:
        test_date_validation()
        test_action_validation()
        test_record_validation()
        test_xml_parsing()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ All tests passed!")
        logger.info("=" * 80)
        return 0
        
    except AssertionError as e:
        logger.error(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
