### block_shock/src/timing_smoke.py
## Toy sleep + CUDA op timing smoke test.

from __future__ import annotations

import time

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency during scaffolding
    torch = None
    _torch_error = exc
else:
    _torch_error = None

from pathlib import Path

from .metrics import TimerRegistry
from .logging_utils import collect_env_info, init_logger, log_metrics


def main() -> None:
    timers = TimerRegistry(sync=True)
    cfg = {
        "run_id": "timing_smoke",
        "phase": {"name": "smoke"},
        "method": {"name": "smoke"},
        "model": {"N": 0, "B": 0, "dtype": "n/a"},
        "hardware": {"world_size": 1},
    }

    sleep_s = 0.05
    with timers.time("sleep"):
        time.sleep(sleep_s)

    print(collect_env_info())

    if torch is None:
        print(f"torch import failed: {_torch_error}")
        print(timers.summary())
        return

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Warm up to avoid first-call overhead dominating.
        x = torch.randn((1024, 1024), device=device)
        _ = x @ x
        torch.cuda.synchronize()

        x = torch.randn((2048, 2048), device=device)
        with timers.time("cuda_gemm"):
            y = x @ x
            _ = y.sum()
    else:
        x = torch.randn((2048, 2048))
        with timers.time("cpu_gemm"):
            y = x @ x
            _ = y.sum()

    summary = timers.summary()
    print(summary)

    run_dir = Path("results/raw") / cfg["run_id"]
    logger = init_logger(cfg, run_dir)
    log_metrics(logger, cfg, {"timings_ms": summary})


if __name__ == "__main__":
    main()
