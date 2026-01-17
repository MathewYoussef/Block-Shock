### block_shock/src/logging_utils.py
## Structured logging to JSONL/CSV.

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import platform
import sys

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
