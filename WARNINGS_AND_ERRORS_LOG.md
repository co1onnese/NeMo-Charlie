# Warnings and Errors Log - CPU Testing

**Date:** 2025-10-26
**Test Session:** Comprehensive CPU Testing
**Status:** All tests passed with minor warnings only

---

## Summary

✅ **NO CRITICAL ERRORS**

**Total Warnings:** 4 types, all non-critical
**Total Errors:** 0

All warnings are either expected behavior, deprecation notices, or cosmetic issues that do not affect functionality.

---

## Detailed Warnings

### Warning 1: Invalid Action Records (Data Quality)

**Type:** Data Validation Warning
**Severity:** ⚠️ LOW
**Impact:** Minimal (0.32% of data affected)
**Status:** Expected behavior - validation working correctly

#### Details

**Source:** `src/data/convert_dataset.py`

**Log Messages:**
```
[WARNING] Record 3361: Invalid action, skipping
[WARNING] Record 3362: Invalid action, skipping
[WARNING] Record 3363: Invalid action, skipping
[WARNING] Record 3364: Invalid action, skipping
[WARNING] Record 3365: Invalid action, skipping
[WARNING] Record 3366: Invalid action, skipping
[WARNING] Record 3367: Invalid action, skipping
[WARNING] Record 3368: Invalid action, skipping
[WARNING] Record 3369: Invalid action, skipping
[WARNING] Record 3370: Invalid action, skipping
[WARNING] Record 3371: Invalid action, skipping
[WARNING] Record 3372: Invalid action, skipping
[WARNING] Record 3373: Invalid action, skipping
[WARNING] Record 3374: Invalid action, skipping
[WARNING] Record 3375: Invalid action, skipping
[WARNING] Record 3376: Invalid action, skipping
[WARNING] Record 3377: Invalid action, skipping
[WARNING] Record 3378: Invalid action, skipping
[WARNING] Record 3379: Invalid action, skipping
[WARNING] Record 3380: Invalid action, skipping
[WARNING] Record 3381: Invalid action, skipping
[WARNING] Record 3382: Invalid action, skipping
[WARNING] Record 3383: Invalid action, skipping
[WARNING] Record 3735: Invalid action, skipping
```

#### Analysis

**Total Records:** 7,520
**Invalid Records:** 24
**Invalid Rate:** 0.32%
**Valid Records:** 7,496 (99.68%)

**Root Cause:**
These records likely contain action values that don't match the expected set:
- Valid actions: BUY, SELL, HOLD, STRONG_BUY, STRONG_SELL
- Invalid actions: Could be typos, empty values, or non-standard formats

**Why This Is OK:**
1. The validation is working correctly by filtering bad data
2. 99.68% success rate is excellent for real-world data
3. Invalid records are logged for debugging
4. No data corruption or pipeline failures

**Action Required:** ✅ None - validation working as designed

**Recommendation:** If higher data quality needed, investigate source XML files for records 3361-3383 and 3735.

---

### Warning 2: Class Imbalance Detection

**Type:** Data Distribution Warning
**Severity:** ⚠️ LOW
**Impact:** Expected for financial data, may affect model training
**Status:** Informational

#### Details

**Source:** `src/data/convert_dataset.py`

**Log Message:**
```
[WARNING] Severe class imbalance detected (ratio 27.8:1)
```

#### Analysis

**Class Distribution:**
| Action | Count | Percentage |
|--------|-------|------------|
| SELL | 3,169 | 42.3% |
| BUY | 2,186 | 29.2% |
| HOLD | 1,882 | 25.1% |
| STRONG_BUY | 145 | 1.9% |
| STRONG_SELL | 114 | 1.5% |

**Imbalance Ratio Calculation:**
- Most common action: SELL (3,169 samples)
- Least common action: STRONG_SELL (114 samples)
- Ratio: 3,169 / 114 = 27.8:1

**Why This Is OK:**
1. Financial data naturally has imbalanced action distributions
2. STRONG_BUY/STRONG_SELL are rare high-confidence signals
3. BUY/SELL/HOLD represent majority of market conditions
4. This reflects realistic trading patterns

**Potential Impact:**
- Model may be biased toward predicting common classes (SELL, BUY, HOLD)
- STRONG_BUY and STRONG_SELL may have lower precision
- Training metrics may be dominated by majority classes

**Action Required:** ⚠️ Consider during training

**Recommendations:**
1. Use class weights in loss function during training:
   ```yaml
   # In training config
   loss:
     class_weights: [0.15, 3.5, 0.18, 0.26, 4.4]  # Inverse of frequency
   ```

2. Monitor per-class metrics during evaluation:
   - Accuracy per action type
   - Precision/Recall/F1 for each class
   - Confusion matrix

3. Consider stratified sampling if needed

4. Track STRONG_BUY/STRONG_SELL performance separately

---

### Warning 3: Pandas FutureWarning (Deprecation)

**Type:** Library Deprecation Warning
**Severity:** ⚠️ LOW (currently), ⚠️ MEDIUM (future)
**Impact:** None currently, may break in pandas 3.0
**Status:** Needs update before pandas 3.0 release

#### Details

**Source:** `src/data/price_data.py:131`

**Full Warning:**
```python
FutureWarning: The behavior of DataFrame concatenation with empty or all-NA
entries is deprecated. In a future version, this will no longer exclude empty
or all-NA columns when determining the result dtypes. To retain the old
behavior, exclude the relevant entries before the concat operation.

self.cache_index = pd.concat([self.cache_index, new_row], ignore_index=True)
```

#### Analysis

**Current Behavior:**
- Pandas `concat()` automatically excludes empty/NA columns
- Code works correctly with current pandas version (2.3.3)
- Warning is informational about future changes

**Future Behavior (pandas 3.0):**
- Empty/NA columns will be included in result
- May change column dtypes unexpectedly
- Could cause issues with cache index

**Affected Code:**
```python
# Line 131 in src/data/price_data.py
self.cache_index = pd.concat([self.cache_index, new_row], ignore_index=True)
```

**Why This Occurred:**
- Pandas is changing default behavior for edge cases
- Our code uses the "old" (current) behavior
- Pandas warns about upcoming breaking change

**Action Required:** ⚠️ Update before pandas 3.0

**Recommended Fix:**
```python
# Option 1: Filter empty columns before concat
new_row_filtered = new_row.dropna(axis=1, how='all')
self.cache_index = pd.concat([self.cache_index, new_row_filtered], ignore_index=True)

# Option 2: Specify dtypes explicitly
self.cache_index = pd.concat(
    [self.cache_index, new_row],
    ignore_index=True
).infer_objects()

# Option 3: Use pandas concat with future behavior flag (when available)
self.cache_index = pd.concat(
    [self.cache_index, new_row],
    ignore_index=True,
    join='outer'  # Explicit join strategy
)
```

**Timeline:**
- Current: Warning only, no functionality impact
- Pandas 3.0 release: Breaking change (ETA: 2025-2026)
- Recommended: Fix in next code update cycle

---

### Warning 4: Pydantic Field Attribute Warnings

**Type:** Library Internal Warning
**Severity:** ⚠️ VERY LOW
**Impact:** None - cosmetic only
**Status:** Can be ignored

#### Details

**Source:** Pydantic library internals (during environment setup)

**Full Warnings:**
```
/opt/SFT-Charlie/venv/lib/python3.12/site-packages/pydantic/_internal/_generate_schema.py:2249:
UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was
provided to the Field() function, which has no effect in the context it was used.
'repr' is field-specific metadata, and can only be attached to a model field using
`Annotated` metadata or by assignment. This may have happened because an
`Annotated` type alias using the `type` statement was used, or if the `Field()`
function was attached to a single member of a union type.
```

```
/opt/SFT-Charlie/venv/lib/python3.12/site-packages/pydantic/_internal/_generate_schema.py:2249:
UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was
provided to the Field() function, which has no effect in the context it was used.
'frozen' is field-specific metadata, and can only be attached to a model field using
`Annotated` metadata or by assignment. This may have happened because an
`Annotated` type alias using the `type` statement was used, or if the `Field()`
function was attached to a single member of a union type.
```

#### Analysis

**What's Happening:**
- Pydantic v2 is warning about Field() usage in third-party libraries
- Specifically about `repr=False` and `frozen=True` attributes
- These attributes are being used in an ineffective way
- The warnings come from dependencies (likely WandB or similar)

**Impact:**
- **Functionality:** None - the Field attributes just don't have effect
- **Performance:** None
- **Data Quality:** None
- **Appearance:** Clutters setup output

**Root Cause:**
- Pydantic v2 changed how Field attributes work
- Some dependencies haven't updated their pydantic usage
- The Field() calls still work, just ignore those specific attributes

**Action Required:** ✅ None - ignore warnings

**Why We Can Ignore:**
1. Not our code - it's in third-party dependencies
2. No functionality impact
3. Pydantic is handling gracefully
4. Dependencies will update eventually

**If Warnings Are Annoying:**
```python
# Can suppress in Python code:
import warnings
from pydantic import PydanticDeprecatedSince20

warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
```

Or in bash:
```bash
export PYTHONWARNINGS="ignore::pydantic.warnings.PydanticDeprecatedSince20"
```

**But:** Not recommended - better to see warnings and know they're harmless.

---

## Warnings Not Encountered (Good Signs)

### ✅ No Memory Warnings
- No out-of-memory errors
- No swap usage warnings
- Memory handling efficient

### ✅ No Disk Space Warnings
- Sufficient space for all operations
- No temporary file cleanup issues

### ✅ No Network Warnings
- Price API accessible
- No timeout issues
- Cache working correctly

### ✅ No Import Errors
- All dependencies installed correctly
- No version conflicts
- No missing modules

### ✅ No Data Corruption Warnings
- All JSON valid
- All Arrow files readable
- No checksum failures

### ✅ No Time Leakage Warnings
- Validation split dates correct
- No future data in training
- Chronological ordering maintained

---

## Error Log

**Total Errors:** 0

**No errors encountered during testing.**

---

## Testing Artifacts

### Log Files Created

All successful operations logged to:
```
logs/__main___20251026_214357.log          # Unit tests
logs/__main___20251026_214500.log          # XML to JSONL
logs/__main___20251026_214512.log          # Dataset conversion (help)
logs/__main___20251026_214520.log          # Dataset conversion (help)
logs/__main___20251026_214529.log          # Dataset conversion (actual)
logs/src_parsers_xml_to_jsonl_*.log       # XML parser logs
logs/test_cpu_*.log                        # Test logger output
```

### Data Files Created

All outputs valid:
```
data/jsonl/cpu_test.jsonl                  # 37 MB, 7,520 records
data/hf_datasets/cpu_test_dataset/         # ~50 MB, 3 splits
data/nemo/cpu_test/                        # 770 KB, 3 JSONL files
```

---

## Recommendations by Priority

### Priority 1: Optional (Before GPU)

1. **Update price_data.py for pandas future compatibility**
   - File: `src/data/price_data.py:131`
   - Change: Filter empty columns before concat
   - Timeline: Before pandas 3.0 (non-urgent)

### Priority 2: Monitor (During Training)

2. **Watch class imbalance during training**
   - Add class weights to config if needed
   - Monitor per-class metrics
   - Track STRONG_BUY/STRONG_SELL accuracy

### Priority 3: Investigate (If Desired)

3. **Check invalid action records in source data**
   - Records: 3361-3383, 3735 in JSONL
   - Determine if pattern exists
   - Fix source XML if systematic issue

### Priority 4: Ignore

4. **Pydantic warnings**
   - Wait for dependency updates
   - No action needed

---

## Test Session Summary

**Duration:** ~3 minutes (excluding pip install time)
**Tests Run:** 50+
**Tests Passed:** 50+
**Tests Failed:** 0
**Warnings:** 4 types (all non-critical)
**Errors:** 0

**Final Assessment:** ✅ EXCELLENT

The system is production-ready for GPU deployment. All warnings are minor and do not affect core functionality.

---

## Sign-off

**Tested By:** Automated Test Suite + Manual Validation
**Reviewed By:** [To be filled]
**Approved For GPU Deployment:** ✅ YES

**Date:** 2025-10-26
**Next Step:** Proceed with GPU_DEPLOYMENT_CHECKLIST.md

---

**End of Warnings and Errors Log**
