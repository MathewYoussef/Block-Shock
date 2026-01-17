### block_shock/src/logging_utils.py
## Structured logging to JSONL/CSV.

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import platform
import sys
import time

#TODO: support CSV aggregation output


def collect_env_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch  # type: ignore

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
            info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:  # pragma: no cover - best-effort metadata
        info["torch_error"] = str(exc)
    return info


def _collect_env_info() -> dict[str, Any]:
    # Backwards-compatible alias.
    return collect_env_info()


def init_logger(cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)

    env_path = run_dir / "env.json"
    env_path.write_text(json.dumps(_collect_env_info(), indent=2, sort_keys=True), encoding="utf-8")

    seed = None
    if isinstance(cfg.get("experiment"), dict):
        seed = cfg["experiment"].get("seed")
    seed_path = run_dir / "seed.txt"
    seed_path.write_text("" if seed is None else str(seed), encoding="utf-8")

    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.touch(exist_ok=True)

    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "env_path": str(env_path),
        "seed_path": str(seed_path),
    }


def log_record(logger: dict[str, Any], record: dict[str, Any]) -> None:
    metrics_path = Path(logger["metrics_path"])
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _get_cuda_peak_bytes() -> int | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _cfg_value(cfg: Mapping[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_metrics_record(
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp_unix": time.time(),
        "run_id": cfg.get("run_id"),
        "phase": _cfg_value(cfg, ["phase", "name"]),
        "method": _cfg_value(cfg, ["method", "name"]),
        "N": _cfg_value(cfg, ["model", "N"]),
        "B": _cfg_value(cfg, ["model", "B"]),
        "dtype": _cfg_value(cfg, ["model", "dtype"]),
        "world_size": _cfg_value(cfg, ["hardware", "world_size"]),
    }

    timings = metrics.get("timings_ms") or metrics.get("timings")
    if timings is not None:
        record["timings_ms"] = timings

    memory_peak = metrics.get("memory_peak_bytes")
    if memory_peak is None:
        memory_peak = _get_cuda_peak_bytes()
    record["memory_peak_bytes"] = memory_peak

    for key, value in metrics.items():
        if key in record or key in ("timings_ms", "timings"):
            continue
        record[key] = value

    if extra:
        record.update(extra)

    return record


def log_metrics(
    logger: dict[str, Any],
    cfg: Mapping[str, Any],
    metrics: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = build_metrics_record(cfg, metrics, extra=extra)
    log_record(logger, record)
    return record
