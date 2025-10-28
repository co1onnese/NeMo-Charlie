#!/usr/bin/env python3
"""
apply_nemo_patches.py
Applies patches to NeMo 2.5.2 to fix optional dependency import errors.

This script patches 6 NeMo files to make nv_one_logger, nemo_run, and tensorstore
imports conditional with graceful degradation.

Usage:
    source venv/bin/activate
    python scripts/apply_nemo_patches.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

def banner(text):
    """Print a formatted banner"""
    print("=" * 70)
    print(text)
    print("=" * 70)

def check_venv():
    """Check if running in a virtual environment"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("[ERROR] Not running in a virtual environment!")
        print("[ERROR] Please activate your venv first:")
        print("[ERROR]   source venv/bin/activate")
        print("[ERROR]   python scripts/apply_nemo_patches.py")
        sys.exit(1)

def find_nemo():
    """Find NeMo installation path"""
    try:
        import nemo
        nemo_path = Path(nemo.__file__).parent
        return nemo_path
    except ImportError:
        print("[ERROR] NeMo not installed in this environment")
        print("[ERROR] Please install NeMo first:")
        print("[ERROR]   pip install nemo-toolkit")
        sys.exit(1)

def backup_file(file_path, backup_dir):
    """Create backup of a file"""
    import shutil
    backup_path = backup_dir / file_path.name
    shutil.copy2(file_path, backup_path)
    return backup_path

def patch_one_logger_callback(file_path):
    """Patch one_logger_callback.py to make nv_one_logger imports conditional"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already patched
    if 'NV_ONE_LOGGER_AVAILABLE' in content:
        return False, "Already patched"

    # Patch 1: Make imports conditional
    old_imports = """from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from nv_one_logger.api.config import OneLoggerConfig
from nv_one_logger.training_telemetry.api.callbacks import on_app_start
from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback

from nemo.lightning.base_callback import BaseCallback"""

    new_imports = """from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

# Conditional import: nv_one_logger packages are optional dependencies
# that are not available on public PyPI. Gracefully degrade if missing.
try:
    from nv_one_logger.api.config import OneLoggerConfig
    from nv_one_logger.training_telemetry.api.callbacks import on_app_start
    from nv_one_logger.training_telemetry.api.config import TrainingTelemetryConfig
    from nv_one_logger.training_telemetry.api.training_telemetry_provider import TrainingTelemetryProvider
    from nv_one_logger.training_telemetry.integration.pytorch_lightning import TimeEventCallback as OneLoggerPTLCallback
    NV_ONE_LOGGER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # Create stub classes when telemetry is unavailable
    NV_ONE_LOGGER_AVAILABLE = False
    OneLoggerConfig = None
    on_app_start = lambda: None
    TrainingTelemetryConfig = None
    TrainingTelemetryProvider = None
    OneLoggerPTLCallback = object  # Stub base class

from nemo.lightning.base_callback import BaseCallback"""

    if old_imports not in content:
        return False, "Import section not found (file may have changed)"

    content = content.replace(old_imports, new_imports)

    # Patch 2: Update __all__
    content = content.replace(
        "__all__ = ['OneLoggerNeMoCallback']",
        "__all__ = ['OneLoggerNeMoCallback', 'NV_ONE_LOGGER_AVAILABLE']"
    )

    # Patch 3: Add availability checks to get_one_logger_init_config
    content = content.replace(
        '''    """
    if "EXP_NAME" in os.environ:
        session_tag = os.environ.get("EXP_NAME")  # For NeMo v1''',
        '''    """
    if not NV_ONE_LOGGER_AVAILABLE:
        return {}

    if "EXP_NAME" in os.environ:
        session_tag = os.environ.get("EXP_NAME")  # For NeMo v1'''
    )

    # Patch 4: Add availability check to _get_base_callback_config
    content = content.replace(
        '''    """
    # Extract values from trainer''',
        '''    """
    if not NV_ONE_LOGGER_AVAILABLE:
        return {}
    # Extract values from trainer'''
    )

    # Patch 5: Create conditional callback class
    # Find the OneLoggerNeMoCallback class definition
    class_start = content.find('class OneLoggerNeMoCallback(')
    if class_start == -1:
        return False, "OneLoggerNeMoCallback class not found"

    # Find the end of the file
    class_end = len(content)

    # Extract everything before the class
    before_class = content[:class_start]

    # Create new class definition
    new_class = '''if NV_ONE_LOGGER_AVAILABLE:
    class OneLoggerNeMoCallback(OneLoggerPTLCallback, BaseCallback):
        """Adapter extending OneLogger's PTL callback with init + config update.

        __init__ configures the provider from meta info, then calls super().__init__.
        update_config computes TrainingTelemetryConfig and applies it.
        """

        _instance = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self) -> None:
            if getattr(self, '_initialized', False):
                return
            init_config = get_one_logger_init_config()
            one_logger_config = OneLoggerConfig(**init_config)
            TrainingTelemetryProvider.instance().with_base_config(
                one_logger_config
            ).with_export_config().configure_provider()
            # Initialize underlying OneLogger PTL callback
            super().__init__(TrainingTelemetryProvider.instance(), call_on_app_start=False)
            # Explicitly signal application start after provider configuration
            on_app_start()
            self._initialized = True

        def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
            # Avoid this function being called multiple times
            if TrainingTelemetryProvider.instance().config.telemetry_config is not None:
                return
            if nemo_version == 'v1':
                config = get_nemo_v1_callback_config(trainer=trainer)
            elif nemo_version == 'v2':
                # v2 expects data module in kwargs
                data = kwargs.get('data', None)
                config = get_nemo_v2_callback_config(trainer=trainer, data=data)
            else:
                config = get_nemo_v1_callback_config(trainer=trainer)
            training_telemetry_config = TrainingTelemetryConfig(**config)
            TrainingTelemetryProvider.instance().set_training_telemetry_config(training_telemetry_config)
else:
    # Create a no-op stub callback when telemetry is unavailable
    class OneLoggerNeMoCallback(BaseCallback):
        """No-op stub for OneLoggerNeMoCallback when nv_one_logger is unavailable."""

        _instance = None

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self) -> None:
            if getattr(self, '_initialized', False):
                return
            super().__init__()
            self._initialized = True

        def update_config(self, nemo_version: str, trainer: Trainer, **kwargs) -> None:
            """No-op update_config for stub callback."""
            pass
'''

    content = before_class + new_class

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def patch_callback_group(file_path):
    """Patch callback_group.py to conditionally instantiate OneLoggerNeMoCallback"""
    with open(file_path, 'r') as f:
        content = f.read()

    if 'NV_ONE_LOGGER_AVAILABLE' in content:
        return False, "Already patched"

    # Patch 1: Update import
    content = content.replace(
        'from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback',
        'from nemo.lightning.one_logger_callback import OneLoggerNeMoCallback, NV_ONE_LOGGER_AVAILABLE'
    )

    # Patch 2: Conditional instantiation
    old_init = '''    def __init__(self) -> None:
        self._callbacks: List[BaseCallback] = [OneLoggerNeMoCallback()]
        # Ensure application-end is emitted at most once per process
        self._app_end_emitted: bool = False'''

    new_init = '''    def __init__(self) -> None:
        # Conditionally add OneLoggerNeMoCallback only if telemetry is available
        # This prevents import errors when nv_one_logger packages are not installed
        if NV_ONE_LOGGER_AVAILABLE:
            self._callbacks: List[BaseCallback] = [OneLoggerNeMoCallback()]
        else:
            self._callbacks: List[BaseCallback] = []
        # Ensure application-end is emitted at most once per process
        self._app_end_emitted: bool = False'''

    if old_init not in content:
        return False, "__init__ method not found"

    content = content.replace(old_init, new_init)

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def patch_gpt_data_api(file_path):
    """Patch gpt/data/api.py to make nemo_run import conditional"""
    with open(file_path, 'r') as f:
        content = f.read()

    if 'NEMO_RUN_AVAILABLE' in content:
        return False, "Already patched"

    old_import = '''import lightning.pytorch as pl
import nemo_run as run

from nemo.collections.llm.gpt.data.dolly import DollyDataModule'''

    new_import = '''import lightning.pytorch as pl

# Conditional import: nemo_run is an optional dependency
try:
    import nemo_run as run
    NEMO_RUN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NEMO_RUN_AVAILABLE = False
    # Create stub decorator when nemo_run is not available
    class _RunStub:
        class cli:
            @staticmethod
            def factory(func):
                return func
        @staticmethod
        def autoconvert(func):
            return func
    run = _RunStub()

from nemo.collections.llm.gpt.data.dolly import DollyDataModule'''

    if old_import not in content:
        return False, "Import section not found"

    content = content.replace(old_import, new_import)

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def patch_model_loader(file_path):
    """Patch model_loader.py to make tensorstore import conditional"""
    with open(file_path, 'r') as f:
        content = f.read()

    if 'TENSORSTORE_AVAILABLE' in content:
        return False, "Already patched"

    old_import = '''import numpy

# tenosrstore is needed to register 'bfloat16' dtype with numpy for zarr compatibility
import tensorstore  # noqa: F401 pylint: disable=unused-import
import torch'''

    new_import = '''import numpy

# tenosrstore is needed to register 'bfloat16' dtype with numpy for zarr compatibility
# Conditional import: tensorstore is an optional dependency
try:
    import tensorstore  # noqa: F401 pylint: disable=unused-import
    TENSORSTORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TENSORSTORE_AVAILABLE = False
    tensorstore = None
import torch'''

    if old_import not in content:
        return False, "Import section not found"

    content = content.replace(old_import, new_import)

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def patch_llm_api(file_path):
    """Patch llm/api.py to make nemo_run import conditional"""
    with open(file_path, 'r') as f:
        content = f.read()

    if 'NEMO_RUN_AVAILABLE' in content:
        return False, "Already patched"

    old_import = '''import lightning.pytorch as pl
import nemo_run as run
import torch'''

    new_import = '''import lightning.pytorch as pl
# Conditional import: nemo_run is an optional dependency
try:
    import nemo_run as run
    NEMO_RUN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NEMO_RUN_AVAILABLE = False
    # Create stub for nemo_run when not available
    def _noop_decorator(*args, **kwargs):
        """No-op decorator that returns the function unchanged"""
        def decorator(func):
            return func
        # Handle both @decorator and @decorator() syntax
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    class _ConfigStub:
        """Stub for run.Config that supports subscripting"""
        def __class_getitem__(cls, item):
            # Return a type that can be used in type annotations
            return type(f'Config[{item}]', (), {})

    class _PartialStub:
        """Stub for run.Partial that supports subscripting"""
        def __class_getitem__(cls, item):
            return type(f'Partial[{item}]', (), {})

    class _RunStub:
        Config = _ConfigStub
        Partial = _PartialStub
        class cli:
            entrypoint = staticmethod(_noop_decorator)
            factory = staticmethod(_noop_decorator)
        autoconvert = staticmethod(_noop_decorator)
    run = _RunStub()
import torch'''

    if old_import not in content:
        return False, "Import section not found"

    content = content.replace(old_import, new_import)

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def patch_llm_init(file_path):
    """Patch llm/__init__.py to export API functions without requiring nemo_run"""
    with open(file_path, 'r') as f:
        content = f.read()

    # Check for a unique marker from our patch
    if '# nemo_run is now handled as an optional dependency by api.py' in content:
        return False, "Already patched"

    old_try_block = '''try:
    import nemo_run as run  # noqa: F401

    from nemo.collections.llm.api import (  # noqa: F401
        distill,
        export_ckpt,
        finetune,
        generate,
        import_ckpt,
        pretrain,
        prune,
        ptq,
        train,
        validate,
    )
    from nemo.collections.llm.recipes import *  # noqa

    __all__.extend(
        [
            "train",
            "import_ckpt",
            "export_ckpt",
            "pretrain",
            "validate",
            "finetune",
            "generate",
            "prune",
            "ptq",
            "distill",
        ]
    )
except ImportError as error:
    logging.warning(f"Failed to import nemo.collections.llm.[api,recipes]: {error}")'''

    new_try_block = '''try:
    # nemo_run is now handled as an optional dependency by api.py
    # so we don't need to import it here first
    from nemo.collections.llm.api import (  # noqa: F401
        distill,
        export_ckpt,
        finetune,
        generate,
        import_ckpt,
        pretrain,
        prune,
        ptq,
        train,
        validate,
    )
    # Only import recipes if nemo_run is available
    try:
        import nemo_run as run  # noqa: F401
        from nemo.collections.llm.recipes import *  # noqa
    except ImportError:
        pass  # recipes unavailable without nemo_run

    __all__.extend(
        [
            "train",
            "import_ckpt",
            "export_ckpt",
            "pretrain",
            "validate",
            "finetune",
            "generate",
            "prune",
            "ptq",
            "distill",
        ]
    )
except ImportError as error:
    logging.warning(f"Failed to import nemo.collections.llm.api functions: {error}")'''

    if old_try_block not in content:
        return False, "Try block not found (file may have changed)"

    content = content.replace(old_try_block, new_try_block)

    with open(file_path, 'w') as f:
        f.write(content)

    return True, "Patched successfully"

def main():
    banner("NeMo 2.5.2 Optional Dependencies Patcher")
    print()
    print("This script patches NeMo to fix import errors for missing optional")
    print("dependencies: nv_one_logger, nemo_run, tensorstore")
    print()

    check_venv()
    nemo_path = find_nemo()

    print(f"[INFO] Virtual environment: {sys.prefix}")
    print(f"[INFO] NeMo installation: {nemo_path}")
    print()

    # Create backup directory
    backup_dir = Path(sys.prefix) / f"nemo_patches_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Creating backups in: {backup_dir}")
    print()

    # Define patches to apply
    patches = [
        ("one_logger_callback.py", nemo_path / "lightning" / "one_logger_callback.py", patch_one_logger_callback),
        ("callback_group.py", nemo_path / "lightning" / "callback_group.py", patch_callback_group),
        ("gpt/data/api.py", nemo_path / "collections" / "llm" / "gpt" / "data" / "api.py", patch_gpt_data_api),
        ("model_loader.py", nemo_path / "export" / "utils" / "model_loader.py", patch_model_loader),
        ("llm/api.py", nemo_path / "collections" / "llm" / "api.py", patch_llm_api),
        ("llm/__init__.py", nemo_path / "collections" / "llm" / "__init__.py", patch_llm_init),
    ]

    banner("Applying Patches")
    print()

    results = []
    for i, (name, file_path, patch_func) in enumerate(patches, 1):
        print(f"[{i}/{len(patches)}] Patching {name}...")

        if not file_path.exists():
            print(f"  ✗ File not found: {file_path}")
            results.append((name, False, "File not found"))
            continue

        # Backup file
        backup_file(file_path, backup_dir)

        # Apply patch
        success, message = patch_func(file_path)

        if success:
            print(f"  ✓ {message}")
        else:
            print(f"  ℹ {message}")

        results.append((name, success, message))
        print()

    # Test the patches
    banner("Testing Patches")
    print()

    try:
        print("[TEST] Importing NeMo LLM...")
        from nemo.collections import llm
        print("  ✓ NeMo LLM imported successfully")

        print("[TEST] Checking import_ckpt...")
        if hasattr(llm, 'import_ckpt'):
            print("  ✓ llm.import_ckpt is available")
        else:
            print("  ✗ llm.import_ckpt NOT available")
            raise Exception("import_ckpt not available")

        print("[TEST] Checking DeepSeek models...")
        if hasattr(llm, 'DeepSeekV3Config') and hasattr(llm, 'DeepSeekModel'):
            print("  ✓ DeepSeek models available")
        else:
            print("  ✗ DeepSeek models NOT available")
            raise Exception("DeepSeek models not available")

        print()
        banner("✓ ALL PATCHES APPLIED SUCCESSFULLY!")
        print()
        print("Your NeMo installation is now patched and working.")
        print("You can run your model conversion with:")
        print("  python scripts/convert/import_to_nemo.py --help")
        print()
        print(f"Backups saved to: {backup_dir}")
        print()
        return 0

    except Exception as e:
        print(f"  ✗ Import test failed: {e}")
        print()
        banner("✗ PATCHING INCOMPLETE")
        print()
        print("Some patches may not have been applied correctly.")
        print(f"Backups available at: {backup_dir}")
        print()
        print("Patch Results:")
        for name, success, message in results:
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {message}")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
