### block_shock/src/orchestrator_smoke.py
## Smoke test for Phase 0 and Phase 1 pipelines using the placeholder method.

from __future__ import annotations

from pathlib import Path

from .logging_utils import init_logger
from .orchestrator import DensePlaceholderMethod, run_phase
from .workloads import build_inputs
from .config import write_config


def _base_cfg(phase_name: str) -> dict:
    return {
        "run_id": f"orchestrator_{phase_name}",
        "model": {"N": 512, "B": 16, "dtype": "float32"},
        "phase": {
            "name": phase_name,
            "warmup_iters": 2,
            "timed_iters": 3,
            "sync_mode": "cuda_events" if phase_name == "phase1_forward" else "sync",
        },
        "workload": {"type": "random_normal", "seed": 1234},
        "hardware": {"world_size": 1},
        "logging": {"out_dir": "results/raw"},
    }


def _run_phase(phase_name: str) -> None:
    cfg = _base_cfg(phase_name)
    inputs = build_inputs(cfg)
    run_dir = Path(cfg["logging"]["out_dir"]) / cfg["run_id"]
    write_config(cfg, run_dir)
    logger = init_logger(cfg, run_dir)
    method = DensePlaceholderMethod()
    result = run_phase(cfg, method, inputs, logger=logger, reference_method=method)
    print(f"{phase_name} results: {result.results}")


def main() -> None:
    _run_phase("phase0_correctness")
    _run_phase("phase1_forward")


if __name__ == "__main__":
    main()
