### block_shock/src/orchestrator.py
## Phase pipelines for phases 0-3.
# Orchestrates build, forward, backward, step, validate, and timing.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None

from .metrics import TimerRegistry
from .logging_utils import log_metrics

#TODO: wire allreduce/comm timing regions where applicable
#TODO: add backward/step pipelines for phases 2-3


@dataclass
class PhaseResult:
    results: dict[str, Any]
    output: Any | None = None


def _compare_tensors(ref, test) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for correctness checks")
    diff = (ref - test).abs()
    max_abs = float(diff.max().item())
    denom = float(ref.abs().max().item())
    rel = max_abs / (denom + 1e-12)
    return {"max_abs_error": max_abs, "rel_error": rel}


def _get_phase_name(cfg: Mapping[str, Any]) -> str:
    phase = cfg.get("phase", {})
    return str(phase.get("name", ""))


def _run_phase0(
    cfg: Mapping[str, Any],
    method: Any,
    inputs: Mapping[str, Any],
    reference_method: Any | None,
    logger: dict[str, Any] | None,
) -> PhaseResult:
    phase = cfg.get("phase", {})
    sync_mode = str(phase.get("sync_mode", "sync"))
    timers = TimerRegistry(sync_mode=sync_mode)

    state = method.build(cfg)
    with timers.time("forward"):
        y_test = method.forward(state, inputs["X"])

    if reference_method is None:
        y_ref = y_test
    else:
        ref_state = reference_method.build(cfg)
        y_ref = reference_method.forward(ref_state, inputs["X"])

    results = _compare_tensors(y_ref, y_test)
    results["phase"] = "phase0_correctness"
    results["timings_ms"] = timers.summary()

    if logger is not None:
        log_metrics(logger, cfg, results)

    return PhaseResult(results=results, output=y_test)


def _run_phase1(
    cfg: Mapping[str, Any],
    method: Any,
    inputs: Mapping[str, Any],
    logger: dict[str, Any] | None,
) -> PhaseResult:
    phase = cfg.get("phase", {})
    sync_mode = str(phase.get("sync_mode", "sync"))
    warmup = int(phase.get("warmup_iters", 5))
    iters = int(phase.get("timed_iters", 10))
    timers = TimerRegistry(sync_mode=sync_mode)

    state = method.build(cfg)
    for _ in range(max(warmup, 0)):
        _ = method.forward(state, inputs["X"])

    for _ in range(max(iters, 1)):
        with timers.time("forward"):
            _ = method.forward(state, inputs["X"])

    results = {
        "phase": "phase1_forward",
        "timings_ms": timers.summary(),
        "iterations": iters,
        "warmup_iters": warmup,
    }
    if logger is not None:
        log_metrics(logger, cfg, results)

    return PhaseResult(results=results, output=None)


class DensePlaceholderMethod:
    def _dtype_from_cfg(self, dtype_name: Any):
        name = str(dtype_name).lower()
        if name in ("bf16", "bfloat16"):
            return torch.bfloat16
        if name in ("fp16", "float16", "half"):
            return torch.float16
        if name in ("fp32", "float32", "float"):
            return torch.float32
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    def build(self, cfg: Mapping[str, Any]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("torch is required for placeholder method")
        n = int(cfg.get("model", {}).get("N", 0))
        if n <= 0:
            raise ValueError("model.N must be set for placeholder method")
        dtype = self._dtype_from_cfg(cfg.get("model", {}).get("dtype", "float32"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w = torch.randn((n, n), device=device, dtype=dtype)
        return {"W": w}

    def forward(self, state: Mapping[str, Any], x):
        return x @ state["W"].t()


def run_phase(
    cfg: Mapping[str, Any],
    method: Any | None,
    inputs: Mapping[str, Any],
    logger: dict[str, Any] | None = None,
    reference_method: Any | None = None,
) -> PhaseResult:
    phase_name = _get_phase_name(cfg)
    if method is None:
        method = DensePlaceholderMethod()

    if phase_name == "phase0_correctness":
        return _run_phase0(cfg, method, inputs, reference_method, logger)
    if phase_name == "phase1_forward":
        return _run_phase1(cfg, method, inputs, logger)

    raise ValueError(f"Unsupported phase: {phase_name}")
