#!/usr/bin/env python3
"""
validate_environment.py - Comprehensive environment validation for NeMo-Charlie

Tests:
  1. Python version and virtual environment
  2. Core dependencies (PyTorch, transformers, etc.)
  3. NeMo Framework and patches
  4. Optional performance packages
  5. GPU/CUDA availability (if not in CPU-only mode)
  6. NeMo LLM functionality
  7. Expected vs unexpected warnings

Usage:
    source venv/bin/activate
    python scripts/validate_environment.py
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

# Suppress warnings during import tests
warnings.filterwarnings('ignore')


class Colors:
    """ANSI color codes"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def banner(text: str):
    """Print formatted banner"""
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)
    print()


def check_status(success: bool, message: str, details: str = ""):
    """Print status message with color coding"""
    if success:
        symbol = f"{Colors.GREEN}✓{Colors.END}"
        print(f"  {symbol} {message}")
        if details:
            print(f"      {details}")
    else:
        symbol = f"{Colors.RED}✗{Colors.END}"
        print(f"  {symbol} {message}")
        if details:
            print(f"      {details}")


def warning_status(message: str, details: str = ""):
    """Print warning message"""
    symbol = f"{Colors.YELLOW}⚠{Colors.END}"
    print(f"  {symbol} {message}")
    if details:
        print(f"      {details}")


def info_status(message: str):
    """Print info message"""
    symbol = f"{Colors.BLUE}ℹ{Colors.END}"
    print(f"  {symbol} {message}")


def check_python_environment() -> bool:
    """Check Python version and virtual environment"""
    banner("Python Environment")
    
    all_ok = True
    
    # Python version
    version = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        check_status(True, f"Python version: {version}")
    else:
        check_status(False, f"Python version: {version}", 
                    "Python 3.10+ required")
        all_ok = False
    
    # Virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        check_status(True, f"Virtual environment: {sys.prefix}")
    else:
        check_status(False, "Virtual environment: NOT ACTIVATED",
                    "Please activate venv: source venv/bin/activate")
        all_ok = False
    
    return all_ok


def check_core_packages() -> bool:
    """Check core ML/data packages"""
    banner("Core Dependencies")
    
    all_ok = True
    packages = [
        ('torch', 'PyTorch', True),
        ('transformers', 'Transformers', True),
        ('datasets', 'HuggingFace Datasets', True),
        ('pandas', 'Pandas', True),
        ('numpy', 'NumPy', True),
        ('omegaconf', 'OmegaConf', True),
        ('lightning', 'Lightning', True),
    ]
    
    for module_name, display_name, required in packages:
        try:
            mod = __import__(module_name)
            version = getattr(mod, '__version__', 'unknown')
            check_status(True, f"{display_name}: {version}")
        except ImportError as e:
            if required:
                check_status(False, f"{display_name}: NOT INSTALLED", str(e))
                all_ok = False
            else:
                warning_status(f"{display_name}: not installed (optional)")
    
    return all_ok


def check_pytorch_cuda() -> Tuple[bool, bool]:
    """Check PyTorch and CUDA availability"""
    banner("PyTorch & CUDA")
    
    pytorch_ok = True
    has_cuda = False
    
    try:
        import torch
        
        # PyTorch version
        version = torch.__version__
        check_status(True, f"PyTorch: {version}")
        
        # CUDA availability
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            check_status(True, f"CUDA: Available (version {cuda_version})")
            check_status(True, f"GPU devices: {device_count}x {device_name}")
            has_cuda = True
        else:
            info_status("CUDA: Not available (CPU-only mode)")
            
        # Quick tensor operation test
        try:
            x = torch.randn(10, 10)
            y = torch.matmul(x, x)
            check_status(True, "PyTorch operations: Working")
        except Exception as e:
            check_status(False, "PyTorch operations: FAILED", str(e))
            pytorch_ok = False
            
    except ImportError as e:
        check_status(False, "PyTorch: NOT INSTALLED", str(e))
        pytorch_ok = False
    
    return pytorch_ok, has_cuda


def check_optional_packages(has_cuda: bool) -> dict:
    """Check optional performance packages"""
    banner("Optional Performance Packages")
    
    results = {}
    
    # zarr - should always be present
    try:
        import zarr
        check_status(True, f"Zarr: {zarr.__version__}",
                    "Checkpoint format support enabled")
        results['zarr'] = True
    except ImportError:
        warning_status("Zarr: not installed",
                      "Install with: pip install zarr>=2.16.0")
        results['zarr'] = False
    
    # transformer-engine - expected in GPU mode
    try:
        import transformer_engine
        version = getattr(transformer_engine, '__version__', 'unknown')
        check_status(True, f"Transformer Engine: {version}",
                    "GPU performance optimizations enabled")
        results['transformer_engine'] = True
    except ImportError:
        if has_cuda:
            warning_status("Transformer Engine: not installed",
                          "GPU detected but TE missing - install for better performance")
        else:
            info_status("Transformer Engine: not installed (not needed in CPU mode)")
        results['transformer_engine'] = False
    
    # flash-attn - optional
    try:
        import flash_attn
        version = getattr(flash_attn, '__version__', 'unknown')
        check_status(True, f"Flash Attention: {version}",
                    "Memory-efficient attention enabled")
        results['flash_attn'] = True
    except ImportError:
        info_status("Flash Attention: not installed (optional)")
        results['flash_attn'] = False
    
    return results


def check_nemo_installation() -> bool:
    """Check NeMo Framework installation"""
    banner("NeMo Framework")
    
    all_ok = True
    
    # NeMo core
    try:
        import nemo
        version = nemo.__version__
        check_status(True, f"NeMo: {version}")
    except ImportError as e:
        check_status(False, "NeMo: NOT INSTALLED", str(e))
        return False
    
    # Megatron-Core
    try:
        import megatron.core
        version = getattr(megatron.core, '__version__', 'unknown')
        check_status(True, f"Megatron-Core: {version}")
    except ImportError as e:
        check_status(False, "Megatron-Core: NOT INSTALLED", str(e))
        all_ok = False
    
    # NVIDIA ModelOpt
    try:
        import modelopt
        version = getattr(modelopt, '__version__', 'unknown')
        check_status(True, f"NVIDIA ModelOpt: {version}")
    except ImportError as e:
        check_status(False, "NVIDIA ModelOpt: NOT INSTALLED", str(e))
        all_ok = False
    
    return all_ok


def check_nemo_patches() -> bool:
    """Check if NeMo patches are applied"""
    banner("NeMo Patches Status")
    
    all_ok = True
    
    try:
        import nemo
        nemo_path = Path(nemo.__file__).parent
        
        patches = [
            ("one_logger_callback.py", nemo_path / "lightning" / "one_logger_callback.py", 
             "NV_ONE_LOGGER_AVAILABLE"),
            ("callback_group.py", nemo_path / "lightning" / "callback_group.py", 
             "NV_ONE_LOGGER_AVAILABLE"),
            ("llm/api.py", nemo_path / "collections" / "llm" / "api.py", 
             "NEMO_RUN_AVAILABLE"),
            ("gpt/data/api.py", nemo_path / "collections" / "llm" / "gpt" / "data" / "api.py", 
             "NEMO_RUN_AVAILABLE"),
            ("model_loader.py", nemo_path / "export" / "utils" / "model_loader.py", 
             "TENSORSTORE_AVAILABLE"),
        ]
        
        for name, file_path, marker in patches:
            if file_path.exists():
                content = file_path.read_text()
                if marker in content:
                    check_status(True, f"{name}: Patched")
                else:
                    check_status(False, f"{name}: NOT PATCHED")
                    all_ok = False
            else:
                check_status(False, f"{name}: File not found")
                all_ok = False
                
    except Exception as e:
        check_status(False, "Patch verification failed", str(e))
        all_ok = False
    
    return all_ok


def check_nemo_llm_functionality() -> Tuple[bool, List[str]]:
    """Check NeMo LLM module functionality and capture warnings"""
    banner("NeMo LLM Functionality")
    
    all_ok = True
    captured_warnings = []
    
    # Capture stderr to check for warnings
    import io
    import contextlib
    
    stderr_capture = io.StringIO()
    
    try:
        with contextlib.redirect_stderr(stderr_capture):
            from nemo.collections import llm
        
        # Check for critical functions
        if hasattr(llm, 'import_ckpt'):
            check_status(True, "llm.import_ckpt: Available")
        else:
            check_status(False, "llm.import_ckpt: NOT AVAILABLE")
            all_ok = False
        
        if hasattr(llm, 'DeepSeekV3Config'):
            check_status(True, "llm.DeepSeekV3Config: Available")
        else:
            check_status(False, "llm.DeepSeekV3Config: NOT AVAILABLE")
            all_ok = False
        
        if hasattr(llm, 'DeepSeekModel'):
            check_status(True, "llm.DeepSeekModel: Available")
        else:
            check_status(False, "llm.DeepSeekModel: NOT AVAILABLE")
            all_ok = False
        
        # Parse captured warnings
        stderr_text = stderr_capture.getvalue()
        if stderr_text:
            for line in stderr_text.split('\n'):
                if 'WARNING' in line or 'Warning' in line:
                    captured_warnings.append(line.strip())
        
    except Exception as e:
        check_status(False, "NeMo LLM import failed", str(e))
        all_ok = False
    
    return all_ok, captured_warnings


def analyze_warnings(warnings: List[str], has_cuda: bool):
    """Analyze and categorize warnings"""
    banner("Warning Analysis")
    
    expected_warnings = {
        'nemo_run': 'NeMo-Run is not installed',
        'zarr': 'Cannot import zarr',
        'transformer_engine': 'transformer_engine not installed',
    }
    
    found_warnings = {key: False for key in expected_warnings}
    unexpected = []
    
    for warning in warnings:
        warning_lower = warning.lower()
        
        if 'nemo' in warning_lower and 'run' in warning_lower:
            found_warnings['nemo_run'] = True
        elif 'zarr' in warning_lower:
            found_warnings['zarr'] = True
        elif 'transformer' in warning_lower and 'engine' in warning_lower:
            found_warnings['transformer_engine'] = True
        else:
            # Ignore common harmless warnings
            if not any(x in warning_lower for x in ['deprecated', 'futurewarning']):
                unexpected.append(warning)
    
    # nemo_run warning is always expected
    if found_warnings['nemo_run']:
        check_status(True, "Expected: NeMo-Run warning present",
                    "This confirms patches are working correctly")
    else:
        warning_status("NeMo-Run warning not seen",
                      "This might mean patches aren't applied, or import path changed")
    
    # zarr warning should NOT appear if installed
    if found_warnings['zarr']:
        check_status(False, "Unexpected: Zarr warning present",
                    "Zarr is installed but NeMo didn't detect it")
    else:
        check_status(True, "No zarr warnings (good)")
    
    # transformer_engine warning depends on mode
    if found_warnings['transformer_engine']:
        if has_cuda:
            warning_status("Transformer Engine warning present",
                          "Consider installing for better GPU performance")
        else:
            check_status(True, "Transformer Engine warning (expected in CPU mode)")
    else:
        check_status(True, "No transformer_engine warnings")
    
    # Report unexpected warnings
    if unexpected:
        print()
        warning_status(f"Found {len(unexpected)} unexpected warnings:")
        for w in unexpected[:5]:  # Show first 5
            print(f"      {w}")
        if len(unexpected) > 5:
            print(f"      ... and {len(unexpected) - 5} more")
    else:
        check_status(True, "No unexpected warnings")


def check_cpu_only_mode() -> bool:
    """Check if CPU_ONLY_MODE is set in .env"""
    try:
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            content = env_file.read_text()
            if 'CPU_ONLY_MODE=true' in content:
                return True
            elif 'CPU_ONLY_MODE=false' in content:
                return False
    except Exception:
        pass
    return True  # Default to CPU mode


def main():
    """Run all validation checks"""
    print(f"{Colors.BOLD}NeMo-Charlie Environment Validation{Colors.END}")
    print(f"Python: {sys.executable}")
    
    cpu_only = check_cpu_only_mode()
    if cpu_only:
        info_status("Mode: CPU-only (from .env)")
    else:
        info_status("Mode: GPU-enabled (from .env)")
    
    # Run all checks
    results = []
    
    results.append(("Python Environment", check_python_environment()))
    results.append(("Core Dependencies", check_core_packages()))
    
    pytorch_ok, has_cuda = check_pytorch_cuda()
    results.append(("PyTorch & CUDA", pytorch_ok))
    
    optional_results = check_optional_packages(has_cuda)
    results.append(("NeMo Installation", check_nemo_installation()))
    results.append(("NeMo Patches", check_nemo_patches()))
    
    llm_ok, warnings = check_nemo_llm_functionality()
    results.append(("NeMo LLM Functionality", llm_ok))
    
    # Analyze warnings
    if warnings:
        analyze_warnings(warnings, has_cuda)
    
    # Final summary
    banner("Validation Summary")
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for category, ok in results:
        check_status(ok, category)
    
    print()
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ ALL CHECKS PASSED ({passed}/{total}){Colors.END}")
        print()
        print("Your environment is ready for:")
        print("  • Model conversion (import_to_nemo.py)")
        print("  • Data processing")
        if has_cuda:
            print("  • GPU-accelerated training")
            print("  • GPU-accelerated inference")
        else:
            print("  • CPU-based operations")
        print()
        
        if not optional_results.get('zarr'):
            print(f"{Colors.YELLOW}Recommendation:{Colors.END}")
            print("  Install zarr to eliminate checkpoint format warning:")
            print("    pip install zarr>=2.16.0")
            print()
        
        if has_cuda and not optional_results.get('transformer_engine'):
            print(f"{Colors.YELLOW}Performance Tip:{Colors.END}")
            print("  Install transformer-engine for better GPU performance:")
            print("    pip install transformer-engine[pytorch]")
            print()
        
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ SOME CHECKS FAILED ({passed}/{total} passed){Colors.END}")
        print()
        print("Please review the errors above and:")
        print("  1. Ensure you ran: bash scripts/setup_env_v2.sh")
        print("  2. Activated venv: source venv/bin/activate")
        print("  3. Check .env file exists and has CPU_ONLY_MODE set")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
