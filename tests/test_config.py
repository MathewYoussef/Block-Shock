### block_shock/tests/test_config.py
## Unit tests for configuration loading and merging.

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

# Import the config module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config as config_utils


def test_load_single_yaml():
    """Test loading a single YAML file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({'test': {'key': 'value'}}, f)
        temp_path = Path(f.name)
    
    try:
        cfg = config_utils.load_yaml(temp_path)
        assert 'test' in cfg
        assert cfg['test']['key'] == 'value'
    finally:
        temp_path.unlink()


def test_merge_configs():
    """Test merging multiple configuration dictionaries."""
    base = {'a': 1, 'b': {'c': 2}}
    override = {'b': {'c': 3, 'd': 4}, 'e': 5}
    
    merged = config_utils.deep_merge(base, override)
    
    assert merged['a'] == 1
    assert merged['b']['c'] == 3
    assert merged['b']['d'] == 4
    assert merged['e'] == 5


def test_merge_preserves_base():
    """Test that merging doesn't modify the base dict."""
    base = {'a': 1, 'b': {'c': 2}}
    base_copy = {'a': 1, 'b': {'c': 2}}
    override = {'b': {'c': 3}}
    
    config_utils.deep_merge(base, override)
    
    # Base should not be modified
    assert base == base_copy


def test_load_multiple_configs():
    """Test loading and merging multiple config files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create base config
        base_path = tmpdir / 'base.yaml'
        with open(base_path, 'w') as f:
            yaml.dump({'model': {'N': 4096}, 'experiment': {'seed': 1234}}, f)
        
        # Create override config
        override_path = tmpdir / 'override.yaml'
        with open(override_path, 'w') as f:
            yaml.dump({'model': {'N': 8192}, 'phase': {'name': 'test'}}, f)
        
        cfg = config_utils.load_and_merge_configs([base_path, override_path])
        
        assert cfg['model']['N'] == 8192  # overridden
        assert cfg['experiment']['seed'] == 1234  # preserved from base
        assert cfg['phase']['name'] == 'test'  # new key


def test_run_id_generation():
    """Test that run IDs are generated with correct format."""
    run_id = config_utils.generate_run_id()
    
    # Check format: YYYYMMDD_HHMMSS_<hash>
    parts = run_id.split('_')
    assert len(parts) == 3
    assert len(parts[0]) == 8  # date
    assert len(parts[1]) == 6  # time
    assert len(parts[2]) == 8  # hash


if __name__ == '__main__':
    # Run tests if pytest is available, otherwise skip
    try:
        pytest.main([__file__, '-v'])
    except ImportError:
        print("pytest not installed. Install with: pip install pytest")
        print("Running basic checks...")
        test_merge_configs()
        test_merge_preserves_base()
        test_run_id_generation()
        print("Basic checks passed!")
