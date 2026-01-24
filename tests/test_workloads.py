### block_shock/tests/test_workloads.py
## Unit tests for workload generation.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the workloads module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src import workloads


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_inputs_gaussian():
    """Test building Gaussian workload."""
    cfg = {
        "model": {"N": 128, "B": 16, "dtype": "float32"},
        "workload": {"type": "random_normal", "mean": 0.0, "std": 1.0, "seed": 42},
        "experiment": {"seed": 42},
    }
    
    result = workloads.build_inputs(cfg)
    
    assert "X" in result
    assert result["X"].shape == (16, 128)
    assert result["X"].dtype == torch.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_inputs_uniform():
    """Test building uniform workload."""
    cfg = {
        "model": {"N": 64, "B": 8, "dtype": "bf16"},
        "workload": {"type": "uniform", "low": -2.0, "high": 2.0, "seed": 123},
    }
    
    result = workloads.build_inputs(cfg)
    
    assert "X" in result
    assert result["X"].shape == (8, 64)
    assert result["X"].dtype == torch.bfloat16
    
    # Check values are in expected range (with some tolerance for fp precision)
    x_float = result["X"].float()
    assert x_float.min() >= -2.1
    assert x_float.max() <= 2.1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_inputs_reproducible():
    """Test that same seed produces same workload."""
    cfg = {
        "model": {"N": 32, "B": 4, "dtype": "float32"},
        "workload": {"type": "random_normal", "seed": 999},
    }
    
    result1 = workloads.build_inputs(cfg)
    result2 = workloads.build_inputs(cfg)
    
    # Same seed should produce identical tensors
    assert torch.allclose(result1["X"], result2["X"])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_inputs_with_target():
    """Test building workload with target tensor."""
    cfg = {
        "model": {"N": 64, "B": 8, "dtype": "float32"},
        "workload": {
            "type": "random_normal",
            "seed": 42,
            "target": "zeros"
        },
    }
    
    result = workloads.build_inputs(cfg)
    
    assert "X" in result
    assert "T" in result
    assert result["T"] is not None
    assert result["T"].shape == result["X"].shape
    assert torch.all(result["T"] == 0.0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_loss_sum():
    """Test sum loss function."""
    cfg = {"workload": {"loss": "sum"}}
    
    loss_fn = workloads.build_loss(cfg)
    
    if TORCH_AVAILABLE:
        y = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = loss_fn(y, None)
        assert loss.item() == 10.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_build_loss_mse():
    """Test MSE loss function."""
    cfg = {"workload": {"loss": "mse"}}
    
    loss_fn = workloads.build_loss(cfg)
    
    if TORCH_AVAILABLE:
        y = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
        loss = loss_fn(y, None)
        assert loss.item() == 4.0  # mean of [4, 4, 4, 4]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_invalid_workload_type():
    """Test that invalid workload type raises error."""
    cfg = {
        "model": {"N": 64, "B": 8, "dtype": "float32"},
        "workload": {"type": "invalid_type"},
    }
    
    with pytest.raises(ValueError, match="Unsupported workload type"):
        workloads.build_inputs(cfg)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_invalid_dtype():
    """Test that invalid dtype raises error."""
    cfg = {
        "model": {"N": 64, "B": 8, "dtype": "invalid_dtype"},
        "workload": {"type": "random_normal"},
    }
    
    with pytest.raises(ValueError, match="Unsupported dtype"):
        workloads.build_inputs(cfg)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_missing_model_params():
    """Test that missing model parameters raise error."""
    cfg = {
        "model": {},  # Missing N and B
        "workload": {"type": "random_normal"},
    }
    
    with pytest.raises(ValueError, match="must be positive integers"):
        workloads.build_inputs(cfg)


if __name__ == '__main__':
    # Run tests if pytest is available
    try:
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        if TORCH_AVAILABLE:
            print("Running basic checks...")
            test_build_inputs_gaussian()
            test_build_inputs_uniform()
            test_build_inputs_reproducible()
            test_build_inputs_with_target()
            test_build_loss_sum()
            test_build_loss_mse()
            print("Basic checks passed!")
        else:
            print("PyTorch not available, skipping tests")
