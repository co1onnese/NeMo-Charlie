# NeMo 2.5.2 Optional Dependencies Fix

## Problem Summary

NeMo 2.5.2 has a critical architectural flaw where optional dependencies are imported **unconditionally at module initialization time**, causing `ModuleNotFoundError` even when these packages are not needed for core functionality.

### Affected Dependencies

1. **`nv_one_logger`** (telemetry) - Three packages:
   - `nv_one_logger_core>=2.3.0`
   - `nv_one_logger_training_telemetry>=2.3.0`
   - `nv_one_logger_pytorch_lightning_integration>=2.3.0`

2. **`nemo_run`** (recipes and CLI utilities)

3. **`tensorstore`** (checkpoint export utilities)

### Why This Fails

1. These packages are marked as **optional** in NeMo's `setup.py` (under `[test]` extras)
2. They **don't exist on public PyPI** (internal NVIDIA packages or not published)
3. They're **not needed** for core LLM tasks like model conversion
4. But NeMo imports them **unconditionally**, causing immediate failure

### Error Symptoms

```
ModuleNotFoundError: No module named 'nv_one_logger'
```

Even when running:
```python
from nemo.collections import llm
```

## Root Cause Analysis

### Import Chain That Fails

```
scripts/convert/import_to_nemo.py
  └─> from nemo.collections import llm
      └─> nemo/collections/llm/__init__.py
          └─> (many intermediate imports)
              └─> nemo/lightning/callback_group.py:21
                  └─> from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback
                      └─> nemo/lightning/one_logger_callback.py:25-29
                          └─> from nv_one_logger.api.config import OneLoggerConfig  # ❌ FAILS
```

Additionally:
- `nemo/collections/llm/gpt/data/api.py` line 16: `import nemo_run as run` ❌
- `nemo/export/utils/model_loader.py` line 25: `import tensorstore` ❌
- `nemo/collections/llm/api.py` line 23: `import nemo_run as run` ❌

### The Design Flaw

**Line 179 of `callback_group.py`**:
```python
CallbackGroup.get_instance()  # Eagerly creates singleton at import time
```

**Line 46 of `callback_group.py`**:
```python
def __init__(self) -> None:
    self._callbacks: List[BaseCallback] = [OneLoggerNeMoCallback()]  # Unconditional instantiation
```

This means the telemetry callback is **always instantiated**, even when:
- Telemetry is not configured
- Telemetry packages are not installed
- User doesn't want telemetry

## Solution Applied

### Files Patched

All patches apply **graceful degradation** - if optional dependencies are missing, NeMo degrades to core functionality without errors.

#### 1. `nemo/lightning/one_logger_callback.py`

**Changes:**
- Wrapped `nv_one_logger` imports in try/except (lines 26-42)
- Added `NV_ONE_LOGGER_AVAILABLE` flag
- Created no-op stub callback when telemetry unavailable (lines 323-343)
- Added availability checks to config functions (lines 60-61, 102-103)

**Result:** Telemetry gracefully disabled when packages missing

#### 2. `nemo/lightning/callback_group.py`

**Changes:**
- Import `NV_ONE_LOGGER_AVAILABLE` flag (line 21)
- Conditional callback instantiation (lines 46-52)

**Result:** Callback group works without telemetry

#### 3. `nemo/collections/llm/gpt/data/api.py`

**Changes:**
- Made `nemo_run` import conditional (lines 17-32)
- Created decorator stubs for `@run.cli.factory`, `@run.autoconvert`

**Result:** Data module API works without nemo_run

#### 4. `nemo/export/utils/model_loader.py`

**Changes:**
- Made `tensorstore` import conditional (lines 25-31)
- Set `TENSORSTORE_AVAILABLE` flag

**Result:** Export utilities work without tensorstore

#### 5. `nemo/collections/llm/api.py`

**Changes:**
- Made `nemo_run` import conditional (lines 23-57)
- Created full stub classes with subscriptable `Config`/`Partial`
- Created no-op decorators for `@run.cli.entrypoint`, etc.

**Result:** Core API functions like `import_ckpt` work without nemo_run

#### 6. `nemo/collections/llm/__init__.py`

**Changes:**
- Removed `nemo_run` pre-import requirement (lines 395-432)
- API functions now export even when nemo_run unavailable
- Recipes import separately within try/except

**Result:** `llm.import_ckpt` and other functions accessible

## Verification

### Test Script

```bash
source venv/bin/activate
python -c "
from nemo.collections import llm
assert hasattr(llm, 'import_ckpt'), 'import_ckpt not available'
assert hasattr(llm, 'DeepSeekV3Config'), 'DeepSeekV3Config not available'
assert hasattr(llm, 'DeepSeekModel'), 'DeepSeekModel not available'
print('✓ All NeMo LLM functions available!')
"
```

### Expected Warnings (Harmless)

```
WARNING: Trying to use Config or Partial, but NeMo-Run is not installed
WARNING: Cannot import zarr, support for zarr-based checkpoints is not available
[NeMo W] Failed to import nemo.collections.llm.[api,recipes]: No module named 'nemo_run'
```

These warnings are **expected** and **harmless** - they indicate optional features are unavailable, but core functionality works.

## Current Status

**✓ PATCHES ALREADY APPLIED** in this environment!

The NeMo installation in `venv/` has been patched and is working correctly.

To verify patches are applied and working:
```bash
source venv/bin/activate
bash scripts/verify_nemo_fixes.sh
```

See `scripts/PATCHES_APPLIED.txt` for details.

## Applying the Fix (When Needed)

You only need to apply patches if you reinstall NeMo or create a new virtual environment.

### Manual Method

If you reinstall NeMo or update it, you'll need to reapply patches. The patches are applied to:

```
$VIRTUAL_ENV/lib/python3.12/site-packages/nemo/
├── lightning/
│   ├── one_logger_callback.py
│   └── callback_group.py
├── collections/llm/
│   ├── __init__.py
│   ├── api.py
│   └── gpt/data/api.py
└── export/utils/
    └── model_loader.py
```

## Impact

### What Works ✅

- ✅ `from nemo.collections import llm`
- ✅ `llm.import_ckpt()` - Model conversion
- ✅ `llm.DeepSeekV3Config` - Model configuration
- ✅ `llm.DeepSeekModel` - Model instantiation
- ✅ All core LLM functionality
- ✅ Model training (telemetry disabled)
- ✅ Model evaluation

### What's Disabled ⚠️

- ⚠️ Telemetry logging (nv_one_logger unavailable)
- ⚠️ NeMo-Run recipes (nemo_run unavailable)
- ⚠️ Zarr checkpoints (zarr unavailable)
- ⚠️ TensorStore export (tensorstore unavailable)

### What's Critical ❌

None - all disabled features are optional!

## Long-term Solution

This is a **NeMo bug** that should be fixed upstream. The proper fix would be:

1. Make all optional dependency imports conditional in NeMo source
2. Publish `nv_one_logger` packages to PyPI if they're required
3. OR make `nv_one_logger` truly optional with lazy initialization

### Reporting to NVIDIA

Consider filing an issue at: https://github.com/NVIDIA/NeMo/issues

**Title:** "NeMo 2.5.2: ModuleNotFoundError for optional dependencies (nv_one_logger, nemo_run, tensorstore)"

**Key points:**
- Optional dependencies imported unconditionally
- Breaks installation from PyPI
- Affects core LLM functionality
- `nv_one_logger` packages don't exist on public PyPI

## Version Compatibility

- **Tested with:** NeMo 2.5.2
- **Python:** 3.12
- **Platform:** Linux

**Note:** If you upgrade NeMo, you'll need to reapply these patches by running:
```bash
bash scripts/fix_nemo_optional_deps.sh
```

## References

- NeMo Installation: https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html
- DeepSeek-V3 Guide: https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/deepseek_v3.html
- Issue date: 2025-10-28
