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


def _compare_tensors(ref, test, eps: float = 1e-12) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for correctness checks")
    diff = (ref - test).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = ref.abs()
    max_rel = float((diff / (denom + eps)).max().item())
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "max_rel_error": max_rel,
    }


def _get_phase_name(cfg: Mapping[str, Any]) -> str:
    phase = cfg.get("phase", {})
    return str(phase.get("name", ""))


def _seed_all(seed: int | None) -> None:
    if seed is None or torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_phase0(
    cfg: Mapping[str, Any],
    method: Any,
    inputs: Mapping[str, Any],
    reference_method: Any | None,
    logger: dict[str, Any] | None,
) -> PhaseResult:
    phase = cfg.get("phase", {})
    sync_mode = str(phase.get("sync_mode", "sync"))
    warmup = int(phase.get("warmup_iters", 10))
    iters = int(phase.get("timed_iters", 100))
    timers = TimerRegistry(sync_mode=sync_mode)

    seed = cfg.get("experiment", {}).get("seed")
    _seed_all(seed)
    state = method.build(cfg)

    if reference_method is None:
        y_ref = None
    else:
        _seed_all(seed)
        ref_state = reference_method.build(cfg)
        y_ref = reference_method.forward(ref_state, inputs["X"])

    for _ in range(max(warmup, 0)):
        _ = method.forward(state, inputs["X"])

    errors_max_abs: list[float] = []
    errors_mean_abs: list[float] = []
    errors_max_rel: list[float] = []
    last_output = None

    for _ in range(max(iters, 1)):
        with timers.time("forward"):
            last_output = method.forward(state, inputs["X"])
        if y_ref is not None:
            eps = float(phase.get("rel_eps", 1e-12))
            metrics = _compare_tensors(y_ref, last_output, eps=eps)
            errors_max_abs.append(metrics["max_abs_error"])
            errors_mean_abs.append(metrics["mean_abs_error"])
            errors_max_rel.append(metrics["max_rel_error"])

    if y_ref is None:
        results = {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "max_rel_error": 0.0,
            "passed": True,
        }
    else:
        tol_max_abs = float(phase.get("tol_max_abs", 0.0))
        tol_max_rel = float(phase.get("tol_max_rel", 0.0))
        passed = (max(errors_max_abs) <= tol_max_abs) and (max(errors_max_rel) <= tol_max_rel)
        results = {
            "max_abs_error": max(errors_max_abs),
            "mean_abs_error": sum(errors_mean_abs) / len(errors_mean_abs),
            "max_rel_error": max(errors_max_rel),
            "passed": passed,
        }

    results["phase"] = "phase0_correctness"
    results["timings_ms"] = timers.summary()
    results["iterations"] = iters
    results["warmup_iters"] = warmup

    if logger is not None:
        log_metrics(logger, cfg, results)

    print(
        "phase0_correctness:",
        f"passed={results['passed']}",
        f"max_abs_error={results['max_abs_error']:.6e}",
        f"max_rel_error={results['max_rel_error']:.6e}",
        f"mean_abs_error={results['mean_abs_error']:.6e}",
        f"warmup_iters={warmup}",
        f"timed_iters={iters}",
        flush=True,
    )

    return PhaseResult(results=results, output=last_output)


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

    forward_stats = results["timings_ms"].get("forward", {})
    print(
        "phase1_forward:",
        f"iterations={iters}",
        f"warmup_iters={warmup}",
        f"forward_avg_ms={forward_stats.get('avg_ms', 0.0):.6f}",
        f"forward_p50_ms={forward_stats.get('p50_ms', 0.0):.6f}",
        f"forward_p95_ms={forward_stats.get('p95_ms', 0.0):.6f}",
        flush=True,
    )

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
        if reference_method is None:
            from .methods import dense_single as reference_method  # local import to avoid cycles
        return _run_phase0(cfg, method, inputs, reference_method, logger)
    if phase_name == "phase1_forward":
        return _run_phase1(cfg, method, inputs, logger)

    raise ValueError(f"Unsupported phase: {phase_name}")
