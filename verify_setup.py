#!/usr/bin/env python3
"""
Setup verification script for Block-Shock.

This script checks that your environment is properly configured to run Block-Shock.
Run this after installing dependencies to verify everything is working.

Usage:
    python verify_setup.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Track what's working
checks = []


def check(name: str, condition: bool, details: str = "") -> bool:
    """Record a check result."""
    checks.append((name, condition, details))
    status = "✓" if condition else "✗"
    print(f"{status} {name}")
    if details and not condition:
        print(f"  {details}")
    return condition


def main():
    """Run setup verification checks."""
    print("=" * 70)
    print("Block-Shock Setup Verification")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # Check Python version
    print("Checking Python environment...")
    py_version = sys.version_info
    py_ok = py_version >= (3, 8)
    check(
        f"Python version {py_version.major}.{py_version.minor}.{py_version.micro}",
        py_ok,
        "Python 3.8 or higher required"
    )
    all_ok = all_ok and py_ok
    print()
    
    # Check PyTorch
    print("Checking PyTorch installation...")
    try:
        import torch
        torch_ok = True
        torch_version = torch.__version__
        check(f"PyTorch {torch_version} installed", True)
        
        # Check CUDA
        cuda_available = torch.cuda.is_available()
        check(f"CUDA available: {cuda_available}", cuda_available,
              "CUDA not available - GPU experiments will not work")
        
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            check(f"GPU count: {gpu_count}", gpu_count > 0,
                  "No GPUs detected")
            
            if gpu_count > 0:
                for i in range(min(gpu_count, 4)):  # Show first 4 GPUs
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"  GPU {i}: {gpu_name}")
        
        # Check distributed
        try:
            import torch.distributed as dist
            check("torch.distributed available", True)
        except ImportError:
            check("torch.distributed available", False,
                  "Distributed training not available")
    
    except ImportError as e:
        torch_ok = False
        check("PyTorch installed", False,
              f"PyTorch not found: {e}")
        all_ok = False
    print()
    
    # Check required packages
    print("Checking required packages...")
    
    try:
        import yaml
        check("pyyaml installed", True)
    except ImportError:
        check("pyyaml installed", False,
              "Install with: pip install pyyaml")
        all_ok = False
    
    try:
        import numpy
        check("numpy installed", True)
    except ImportError:
        check("numpy installed", False,
              "Install with: pip install numpy")
        all_ok = False
    
    try:
        import matplotlib
        check("matplotlib installed", True)
    except ImportError:
        check("matplotlib installed", False,
              "Install with: pip install matplotlib")
        all_ok = False
    
    print()
    
    # Check optional packages
    print("Checking optional packages...")
    
    try:
        import pytest
        check("pytest installed (for testing)", True)
    except ImportError:
        check("pytest installed (optional)", False,
              "Install with: pip install pytest")
    
    print()
    
    # Check project structure
    print("Checking project structure...")
    project_root = Path(__file__).parent
    
    dirs_to_check = [
        "src",
        "configs",
        "examples",
        "tests",
        "analysis",
        "scripts",
    ]
    
    for dirname in dirs_to_check:
        dir_path = project_root / dirname
        check(f"Directory '{dirname}' exists", dir_path.exists(),
              f"Expected directory at {dir_path}")
    
    print()
    
    # Check src imports
    print("Checking Block-Shock modules...")
    sys.path.insert(0, str(project_root))
    
    modules = [
        "src.config",
        "src.distributed",
        "src.logging_utils",
        "src.metrics",
        "src.workloads",
    ]
    
    for module_name in modules:
        try:
            __import__(module_name)
            check(f"Import {module_name}", True)
        except Exception as e:
            check(f"Import {module_name}", False, str(e))
            all_ok = False
    
    print()
    
    # Summary
    print("=" * 70)
    if all_ok:
        print("✓ All critical checks passed!")
        print()
        print("Your environment is ready to run Block-Shock.")
        print()
        print("Next steps:")
        print("  1. Try the examples: cd examples && python simple_correctness_check.py")
        print("  2. Read the documentation: less README.md")
        print("  3. Run tests: pytest tests/")
    else:
        print("✗ Some checks failed.")
        print()
        print("Please fix the issues above before running Block-Shock.")
        print("See README.md for installation instructions.")
    print("=" * 70)
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
