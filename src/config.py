### block_shock/src/config.py
## YAML config loading and merging utilities.

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import datetime as dt
import uuid

import yaml


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary.
    
    Args:
        path: Path to the YAML file to load
        
    Returns:
        Dictionary containing the YAML file contents
        
    Raises:
        ValueError: If the YAML root is not a mapping
    """
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


# Alias for backward compatibility
load_yaml = load_yaml_file


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.
    
    Later values (from override) take precedence over earlier ones (from base).
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base dictionary
        override: Dictionary with override values
        
    Returns:
        New merged dictionary (does not modify inputs)
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_yaml_files(paths: Iterable[Path]) -> dict[str, Any]:
    """Load and merge multiple YAML files in order.
    
    Files are merged left-to-right, with later files overriding earlier ones.
    
    Args:
        paths: Iterable of paths to YAML files
        
    Returns:
        Merged configuration dictionary
    """
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml_file(path))
    return merged


# Alias for backward compatibility
load_and_merge_configs = merge_yaml_files


def generate_run_id() -> str:
    """Generate a unique run identifier.
    
    Format: YYYYMMDD_HHMMSS_<8-char-hex>
    Example: 20260124_143022_a1b2c3d4
    
    Returns:
        Unique run identifier string
    """
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}_{suffix}"


def generate_run_group() -> str:
    """Generate a run group identifier for organizing related runs.
    
    Format: YYYYMMDD_HHMMSS
    Example: 20260124_143022
    
    Returns:
        Run group identifier string
    """
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_config(paths: Iterable[Path], run_id: str | None = None) -> dict[str, Any]:
    """Resolve final configuration by merging files and adding metadata.
    
    Merges all config files in order, then adds:
    - Auto-generated run_group if auto_group is enabled
    - Run ID (auto-generated or provided)
    - Config file paths
    
    Args:
        paths: Paths to config YAML files to merge
        run_id: Optional explicit run ID (auto-generated if None)
        
    Returns:
        Resolved configuration dictionary with metadata
    """
    merged = merge_yaml_files(paths)
    logging_cfg = dict(merged.get("logging", {}) or {})
    if logging_cfg.get("auto_group") and not logging_cfg.get("run_group"):
        logging_cfg["run_group"] = generate_run_group()
    merged["logging"] = logging_cfg
    merged["run_id"] = run_id or generate_run_id()
    merged["config_paths"] = [str(p) for p in paths]
    return merged


def config_to_yaml(config: dict[str, Any]) -> str:
    """Convert configuration dictionary to YAML string.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        YAML-formatted string
    """
    return yaml.safe_dump(config, sort_keys=False)


def write_config(config: dict[str, Any], run_dir: Path) -> Path:
    """Write configuration to a YAML file in the run directory.
    
    Creates the run directory if it doesn't exist.
    
    Args:
        config: Configuration dictionary to write
        run_dir: Directory where the config file will be written
        
    Returns:
        Path to the written config.yaml file
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "config.yaml"
    out_path.write_text(config_to_yaml(config), encoding="utf-8")
    return out_path
