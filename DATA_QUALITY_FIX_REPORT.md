# Data Quality Fix Report

**Date:** 2025-10-26
**Issue:** Invalid action records causing data loss
**Status:** ✅ **RESOLVED**

---

## Problem Statement

During initial CPU testing, 24 records (0.32% of data) were being skipped with warnings:
```
[WARNING] Record 3361: Invalid action, skipping
[WARNING] Record 3362: Invalid action, skipping
... (22 more warnings)
```

This was causing confusion and potential data loss, as valid-looking records were being filtered out late in the pipeline.

---

## Root Cause Analysis

### Investigation Findings

1. **Source of Issue:** Records in source XML files with `action='error'`
2. **Reason for Errors:** API quota exhaustion during thesis generation (Error code: 402 - Insufficient Balance)
3. **Affected Data:**
   - **Tickers:** TGT (23 records), TSLA (1 record)
   - **Date Range:** 2025-03-20 to 2025-04-24
   - **Total:** 24 records out of 7,520 (0.32%)

### Example Error Record

**XML Content:**
```xml
<thesis>
  <as-of-date>2025-03-24</as-of-date>
  <reasoning>ERROR: Error code: 402 - {'error': {'message': 'Insufficient Balance',
             'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}</reasoning>
  <action>error</action>
  <support>Failed to generate thesis due to API error</support>
</thesis>
```

### Pipeline Behavior

**Before Fix:**
- XML Parser: Processed all 7,520 records (including errors)
- Dataset Converter: Filtered out 24 error records
- **Problem:** User didn't know about bad data until dataset conversion

**After Fix:**
- XML Parser: Filters error records upfront, reports 24 skipped
- Dataset Converter: Processes all 7,496 valid records
- **Benefit:** Clear visibility into data quality from the start

---

## Solution Implemented

### Code Changes

**File:** `src/parsers/xml_to_jsonl.py`

#### 1. Added Command-Line Flag
```python
parser.add_argument("--keep_errors", action="store_true", default=False,
                   help="Keep records with action='error'")
```

**Use Case:** Debugging or analyzing API failures

#### 2. Error Filtering Logic
```python
# Filter out error records unless --keep_errors flag is set
if r.get("action") == "error" and not args.keep_errors:
    skipped_errors += 1
    if skipped_errors <= 5:  # Log first 5
        logger.warning(f"Skipping error record: {r.get('ticker')} on {r.get('as_of_date')} (action='error')")
    continue
```

**Benefits:**
- Filters bad data at source
- Logs first 5 for visibility
- Counts total for reporting

#### 3. Enhanced Reporting
```python
if skipped_errors > 0:
    logger.info(f"Skipped {skipped_errors} records with action='error' (API failures)")
    logger.info(f"  (Use --keep_errors flag to include these records)")
logger.info(f"Parse errors: {errors}")
```

**Benefits:**
- Clear separation: skipped errors vs parse errors
- User knows exactly what happened
- Instructions for including errors if needed

---

## Validation Results

### Before Fix

```
Pipeline Run:
  XML → JSONL:           7,520 records
  JSONL → HF Dataset:    7,496 records ⚠️ (24 skipped)

Warnings Generated:    24 confusing warnings during dataset conversion
Data Visibility:       Poor - user doesn't know about bad data upfront
```

### After Fix

```
Pipeline Run:
  XML → JSONL:           7,496 records (24 filtered, clearly reported)
  JSONL → HF Dataset:    7,496 records ✅ (0 skipped)

Warnings Generated:    0 during dataset conversion
Data Visibility:       Excellent - clear reporting at XML parsing stage
```

### Complete Pipeline Test

```
Step 1: XML → JSONL
  ✅ Wrote 7,496 records to data/jsonl/production.jsonl
  ✅ Skipped 24 records with action='error' (API failures)
  ✅ Parse errors: 0

Step 2: JSONL → HF Dataset
  ✅ Loaded 7,496 records from data/jsonl/production.jsonl
  ✅ Normalized 7,496 records, skipped 0
  ✅ Train: 5,560, Validation: 420, Test: 1,516

Step 3: HF → NeMo JSONL
  ✅ training.jsonl, validation.jsonl, test.jsonl created
  ✅ All splits exported successfully
```

---

## Data Quality Metrics

### Final Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Raw Records** | 7,520 |
| **API Error Records** | 24 (0.32%) |
| **Valid Records** | 7,496 (99.68%) |
| **Data Quality Score** | 99.68% ✅ |

### Action Distribution (Valid Records Only)

| Action | Count | Percentage |
|--------|-------|------------|
| SELL | 3,169 | 42.3% |
| BUY | 2,186 | 29.2% |
| HOLD | 1,882 | 25.1% |
| STRONG_BUY | 145 | 1.9% |
| STRONG_SELL | 114 | 1.5% |
| **Total** | **7,496** | **100%** |

### Time-Based Splits

| Split | Samples | Percentage | Date Range |
|-------|---------|------------|------------|
| Train | 5,560 | 74.2% | up to 2024-12-01 |
| Validation | 420 | 5.6% | 2024-12-01 to 2024-12-31 |
| Test | 1,516 | 20.2% | from 2025-01-01 |
| **Total** | **7,496** | **100%** | - |

---

## Usage Guide

### Standard Usage (Recommended)

```bash
# Default behavior: filter out error records
python3 src/parsers/xml_to_jsonl.py \
    --input_dir data/raw_xml \
    --output_file data/jsonl/production.jsonl
```

**Output:**
```
Wrote 7,496 records to data/jsonl/production.jsonl
Skipped 24 records with action='error' (API failures)
Parse errors: 0
```

### Debugging Mode (Optional)

```bash
# Keep error records for analysis
python3 src/parsers/xml_to_jsonl.py \
    --input_dir data/raw_xml \
    --output_file data/jsonl/with_errors.jsonl \
    --keep_errors
```

**Output:**
```
Wrote 7,520 records to data/jsonl/with_errors.jsonl
Parse errors: 0
```

**Use Cases:**
- Analyzing API failure patterns
- Debugging thesis generation system
- Understanding when/why API failures occurred

---

## Impact Assessment

### Benefits

1. **✅ Data Quality Visibility**
   - Users know immediately how many records are bad
   - Clear distinction between parse errors and API failures
   - No surprises later in the pipeline

2. **✅ Pipeline Efficiency**
   - Bad data filtered once, at the source
   - Downstream components process only valid data
   - No wasted processing on error records

3. **✅ Debugging Support**
   - `--keep_errors` flag for analysis
   - First 5 errors logged for quick inspection
   - Total count always reported

4. **✅ Production Readiness**
   - Clean, validated data from the start
   - 100% of valid data processed
   - No unexpected data loss

### Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Invalid Action Warnings | 24 | 0 | ✅ 100% reduction |
| Data Loss Visibility | Poor | Excellent | ✅ Major improvement |
| Pipeline Stages with Filtering | 2 | 1 | ✅ Simplified |
| User Confusion | High | Low | ✅ Much clearer |

---

## Testing Performed

### Unit Tests
- ✅ XML parsing with error records
- ✅ Filtering logic validation
- ✅ Flag behavior (--keep_errors)
- ✅ Statistics reporting

### Integration Tests
- ✅ Complete pipeline: XML → JSONL → HF → NeMo
- ✅ Data quality validation at each stage
- ✅ Time-based split integrity
- ✅ No data leakage between splits

### Edge Cases Tested
- ✅ All error records (--keep_errors)
- ✅ No error records (normal case)
- ✅ Mixed valid/error records
- ✅ Error records at start/middle/end of file

---

## Recommendations

### For Production Use

1. **Use Default Behavior**
   ```bash
   python3 src/parsers/xml_to_jsonl.py \
       --input_dir data/raw_xml \
       --output_file data/jsonl/production.jsonl
   ```
   - Automatically filters error records
   - Clean data for training
   - Clear reporting

2. **Monitor Error Counts**
   - Check log messages for skipped error count
   - Investigate if count is unexpectedly high
   - Contact thesis generation team if >1% errors

3. **Document Data Quality**
   - Record the number of skipped errors
   - Include in training manifest
   - Track over time for quality trends

### For Debugging

1. **Analyze Error Patterns**
   ```bash
   python3 src/parsers/xml_to_jsonl.py \
       --input_dir data/raw_xml \
       --output_file data/jsonl/with_errors.jsonl \
       --keep_errors

   # Then analyze error records
   grep '"action": "error"' data/jsonl/with_errors.jsonl | jq .
   ```

2. **Identify Affected Tickers**
   ```bash
   grep '"action": "error"' data/jsonl/with_errors.jsonl | \
       jq -r '.ticker' | sort | uniq -c
   ```

3. **Check Date Distribution**
   ```bash
   grep '"action": "error"' data/jsonl/with_errors.jsonl | \
       jq -r '.as_of_date' | sort
   ```

---

## Lessons Learned

### What Went Well

1. **Early Detection:** Testing caught the issue before production
2. **Root Cause Analysis:** Quickly identified source XML as the problem
3. **Clean Solution:** Simple, effective filtering at the right place
4. **User Experience:** Clear reporting and optional flag for flexibility

### What Could Be Improved

1. **Upstream Prevention:** Could add validation to thesis generation to prevent API errors from being saved
2. **Automated Alerts:** Could add monitoring to alert when error rate exceeds threshold
3. **Recovery Logic:** Could add retry logic to thesis generation for transient API failures

### Best Practices Reinforced

1. **Filter at Source:** Remove bad data as early as possible
2. **Log Clearly:** Make it obvious what's happening and why
3. **Provide Options:** Give users control with flags when appropriate
4. **Test Thoroughly:** Run complete pipeline after fixes

---

## Conclusion

✅ **Issue Fully Resolved**

The data quality issue has been completely fixed. The pipeline now:
- Filters error records during XML parsing
- Provides clear visibility into data quality
- Processes 100% of valid data without warnings
- Offers debugging support via `--keep_errors` flag

**Final Data Quality:** 99.68% valid records (7,496 out of 7,520)

**Production Readiness:** ✅ Ready for GPU deployment

---

## Appendix: Error Record Details

### Affected Tickers

| Ticker | Error Count | Date Range |
|--------|-------------|------------|
| TGT | 23 | 2025-03-24 to 2025-04-24 |
| TSLA | 1 | 2025-03-20 |
| **Total** | **24** | 2025-03-20 to 2025-04-24 |

### Error Message Pattern

All 24 records contain the same error message:
```
ERROR: Error code: 402 - {'error': {'message': 'Insufficient Balance',
'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}
```

This indicates an API quota/billing issue during thesis generation for these specific dates.

### Recommended Action for Thesis Generation Team

1. Review API quota/billing for dates: March 20 - April 24, 2025
2. Regenerate theses for these 24 missing records if needed
3. Implement retry logic for API quota errors
4. Add monitoring to alert on API failures before they accumulate

---

**Report Prepared By:** Automated Testing & Quality Assurance
**Date:** 2025-10-26
**Status:** Issue Closed ✅
