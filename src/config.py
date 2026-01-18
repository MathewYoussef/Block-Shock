### block_shock/src/config.py
## YAML config loading and merging utilities.

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import datetime as dt
import uuid

import yaml


def load_yaml_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def merge_yaml_files(paths: Iterable[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, load_yaml_file(path))
    return merged


def generate_run_id() -> str:
    stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{stamp}_{suffix}"


def generate_run_group() -> str:
    return dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_config(paths: Iterable[Path], run_id: str | None = None) -> dict[str, Any]:
    merged = merge_yaml_files(paths)
    logging_cfg = dict(merged.get("logging", {}) or {})
    if logging_cfg.get("auto_group") and not logging_cfg.get("run_group"):
        logging_cfg["run_group"] = generate_run_group()
    merged["logging"] = logging_cfg
    merged["run_id"] = run_id or generate_run_id()
    merged["config_paths"] = [str(p) for p in paths]
    return merged


def config_to_yaml(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False)


def write_config(config: dict[str, Any], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "config.yaml"
    out_path.write_text(config_to_yaml(config), encoding="utf-8")
    return out_path
