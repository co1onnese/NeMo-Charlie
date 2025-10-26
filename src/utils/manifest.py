"""
manifest.py
Generate reproducibility manifests for training runs.
Records git commit, config hashes, data checksums, environment info.
"""
import hashlib
import json
import os
import sys
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess


def get_git_info(repo_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get git repository information.
    
    Args:
        repo_path: Path to git repository (default: current directory)
        
    Returns:
        Dictionary with git info (commit hash, branch, remote, dirty status)
    """
    if repo_path is None:
        repo_path = os.getcwd()
    
    git_info = {
        "commit": None,
        "branch": None,
        "remote": None,
        "dirty": None,
        "error": None
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["commit"] = result.stdout.strip()
        
        # Get branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()
        
        # Get remote
        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["remote"] = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0
            
    except Exception as e:
        git_info["error"] = str(e)
    
    return git_info


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256, etc.)
        
    Returns:
        Hex digest of file hash
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SHA256 hash of config
    """
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()


def get_environment_info() -> Dict[str, Any]:
    """
    Collect environment information.
    
    Returns:
        Dictionary with Python version, platform, packages
    """
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "packages": {}
    }
    
    # Try to get key package versions
    try:
        import torch
        env_info["packages"]["torch"] = torch.__version__
    except ImportError:
        pass
    
    try:
        import transformers
        env_info["packages"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import datasets
        env_info["packages"]["datasets"] = datasets.__version__
    except ImportError:
        pass
    
    return env_info


def create_manifest(
    run_name: str,
    config: Dict[str, Any],
    data_files: Optional[Dict[str, str]] = None,
    output_path: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive reproducibility manifest.
    
    Args:
        run_name: Name of this training run
        config: Configuration dictionary
        data_files: Dictionary of data file paths to hash
        output_path: Path to save manifest JSON (optional)
        additional_info: Any additional information to include
        
    Returns:
        Manifest dictionary
    """
    manifest = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "git": get_git_info(os.environ.get("GIT_REPO")),
        "environment": get_environment_info(),
        "config": config,
        "config_hash": compute_config_hash(config),
        "data_files": {}
    }
    
    # Hash data files
    if data_files:
        for name, path in data_files.items():
            if os.path.exists(path):
                try:
                    manifest["data_files"][name] = {
                        "path": path,
                        "hash": compute_file_hash(path),
                        "size": os.path.getsize(path)
                    }
                except Exception as e:
                    manifest["data_files"][name] = {
                        "path": path,
                        "error": str(e)
                    }
    
    # Add any additional info
    if additional_info:
        manifest["additional"] = additional_info
    
    # Save to file
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
    
    return manifest
