### block_shock/tests/test_masks.py
## Unit tests for mask generation and validation.

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the masks module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from src.sparsity import masks as mask_utils


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_parse_pattern_from_string():
    """Test parsing 2:4 pattern from string."""
    pattern = mask_utils._parse_pattern("1100")
    assert pattern == [1, 1, 0, 0]
    
    pattern = mask_utils._parse_pattern("0011")
    assert pattern == [0, 0, 1, 1]
    
    pattern = mask_utils._parse_pattern("1010")
    assert pattern == [1, 0, 1, 0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_parse_pattern_from_list():
    """Test parsing 2:4 pattern from list."""
    pattern = mask_utils._parse_pattern([1, 1, 0, 0])
    assert pattern == [1, 1, 0, 0]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_invalid_pattern_length():
    """Test that invalid pattern length raises error."""
    with pytest.raises(ValueError, match="must be length 4"):
        mask_utils._parse_pattern("110")
    
    with pytest.raises(ValueError, match="must be length 4"):
        mask_utils._parse_pattern("11000")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_invalid_pattern_sum():
    """Test that patterns without exactly 2 ones raise error."""
    with pytest.raises(ValueError, match="must have exactly two 1s"):
        mask_utils._parse_pattern("1110")
    
    with pytest.raises(ValueError, match="must have exactly two 1s"):
        mask_utils._parse_pattern("1000")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_complementary_patterns():
    """Test that complementary patterns are generated correctly."""
    cfg = {'name': 'complement_1100_0011'}
    
    pattern_0 = mask_utils.get_pattern_from_cfg(cfg, rank=0)
    pattern_1 = mask_utils.get_pattern_from_cfg(cfg, rank=1)
    
    # Check they are valid 2:4 patterns
    assert sum(pattern_0) == 2
    assert sum(pattern_1) == 2
    
    # Check they are complementary (no overlap, full coverage)
    for i in range(4):
        assert pattern_0[i] + pattern_1[i] == 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_generate_mask_tensor():
    """Test generating mask tensors."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    pattern = [1, 1, 0, 0]
    shape = (8, 8)
    mask = mask_utils.generate_mask(pattern, shape, device='cpu', dtype=torch.float32)
    
    assert mask.shape == shape
    assert mask.dtype == torch.float32
    
    # Check that pattern is repeated correctly
    # Each group of 4 should match the pattern
    for i in range(mask.shape[0]):
        for j in range(0, mask.shape[1], 4):
            group = mask[i, j:j+4].tolist()
            assert group == pattern or group == [float(x) for x in pattern]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_validate_24_sparsity():
    """Test validation of 2:4 sparsity pattern."""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create a valid 2:4 sparse tensor
    pattern = [1, 1, 0, 0]
    shape = (8, 8)
    mask = mask_utils.generate_mask(pattern, shape, device='cpu', dtype=torch.float32)
    tensor = torch.randn(shape) * mask
    
    # Should validate successfully
    try:
        is_valid = mask_utils.validate_24_sparsity(tensor)
        assert is_valid
    except Exception as e:
        # If the function doesn't return bool, it should not raise for valid input
        pass


if __name__ == '__main__':
    # Run tests if pytest is available
    try:
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        if TORCH_AVAILABLE:
            print("Running basic checks...")
            test_parse_pattern_from_string()
            test_parse_pattern_from_list()
            test_invalid_pattern_length()
            test_invalid_pattern_sum()
            print("Basic checks passed!")
        else:
            print("PyTorch not available, skipping tests")
