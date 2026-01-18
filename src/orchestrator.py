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

from .metrics import TimerRegistry, summarize_samples_ms
from .logging_utils import log_metrics
from . import distributed as dist_utils
from . import workloads

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


def _reset_allreduce_samples(state: Mapping[str, Any]) -> None:
    if not isinstance(state, dict):
        return
    if "allreduce_event_pairs" in state:
        state["allreduce_event_pairs"] = []
    if "allreduce_samples_ms" in state:
        state["allreduce_samples_ms"] = []
    if "layout_fix_event_pairs" in state:
        state["layout_fix_event_pairs"] = []
    if "layout_fix_samples_ms" in state:
        state["layout_fix_samples_ms"] = []
    if "layout_fix_did_copy" in state:
        state["layout_fix_did_copy"] = []
    if "layout_fix_bytes" in state:
        state["layout_fix_bytes"] = []


def _collect_allreduce_samples_ms(state: Mapping[str, Any]) -> list[float]:
    samples: list[float] = []
    if isinstance(state, Mapping):
        samples.extend(list(state.get("allreduce_samples_ms", [])))
        event_pairs = list(state.get("allreduce_event_pairs", []))
        if event_pairs and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            for start_evt, end_evt in event_pairs:
                samples.append(float(start_evt.elapsed_time(end_evt)))
    return samples


def _attach_allreduce_timings(results: dict[str, Any], state: Mapping[str, Any]) -> None:
    samples_ms = _collect_allreduce_samples_ms(state)
    if not samples_ms:
        return
    timings = results.setdefault("timings_ms", {})
    timings["allreduce"] = summarize_samples_ms(samples_ms)


def _collect_layout_fix_samples_ms(state: Mapping[str, Any]) -> list[float]:
    samples: list[float] = []
    if isinstance(state, Mapping):
        samples.extend(list(state.get("layout_fix_samples_ms", [])))
        event_pairs = list(state.get("layout_fix_event_pairs", []))
        if event_pairs and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            for start_evt, end_evt in event_pairs:
                samples.append(float(start_evt.elapsed_time(end_evt)))
    return samples


def _attach_layout_fix_metrics(results: dict[str, Any], state: Mapping[str, Any]) -> None:
    samples_ms = _collect_layout_fix_samples_ms(state)
    if samples_ms:
        timings = results.setdefault("timings_ms", {})
        timings["layout_fix"] = summarize_samples_ms(samples_ms)

    if not isinstance(state, Mapping):
        return
    did_copy = list(state.get("layout_fix_did_copy", []))
    bytes_list = list(state.get("layout_fix_bytes", []))
    if not did_copy:
        return
    total_iters = len(did_copy)
    copies = sum(1 for flag in did_copy if flag)
    bytes_total = sum(int(val) for val in bytes_list)
    results["layout_fix_trigger_rate"] = copies / total_iters
    results["layout_fix_bytes_per_iter"] = bytes_total / total_iters
    results["layout_fix_bytes_per_copy"] = (bytes_total / copies) if copies else 0.0


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

    if "W_full" in state:
        if torch is None:
            raise RuntimeError("torch is required for reference output")
        y_ref = torch.nn.functional.linear(inputs["X"], state["W_full"], state.get("bias"))
    elif reference_method is None:
        y_ref = None
    else:
        _seed_all(seed)
        ref_state = reference_method.build(cfg)
        y_ref = reference_method.forward(ref_state, inputs["X"])

    if y_ref is not None and dist_utils.is_distributed():
        dist_utils.broadcast_tensor(y_ref, src=0)

    for _ in range(max(warmup, 0)):
        _ = method.forward(state, inputs["X"])

    _reset_allreduce_samples(state)

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
    _attach_allreduce_timings(results, state)
    _attach_layout_fix_metrics(results, state)
    results["iterations"] = iters
    results["warmup_iters"] = warmup

    if logger is not None:
        log_metrics(logger, cfg, results)

    if dist_utils.rank() == 0:
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

    _reset_allreduce_samples(state)

    for _ in range(max(iters, 1)):
        with timers.time("forward"):
            _ = method.forward(state, inputs["X"])

    results = {
        "phase": "phase1_forward",
        "timings_ms": timers.summary(),
        "iterations": iters,
        "warmup_iters": warmup,
    }
    _attach_allreduce_timings(results, state)
    _attach_layout_fix_metrics(results, state)
    if logger is not None:
        log_metrics(logger, cfg, results)

    forward_stats = results["timings_ms"].get("forward", {})
    if dist_utils.rank() == 0:
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


def _run_phase2(
    cfg: Mapping[str, Any],
    method: Any,
    inputs: Mapping[str, Any],
    reference_method: Any | None,
    logger: dict[str, Any] | None,
) -> PhaseResult:
    if torch is None:
        raise RuntimeError("torch is required for phase 2")
    phase = cfg.get("phase", {})
    sync_mode = str(phase.get("sync_mode", "sync"))
    warmup = int(phase.get("warmup_iters", 10))
    iters = int(phase.get("timed_iters", 100))
    timers = TimerRegistry(sync_mode=sync_mode)
    loss_fn = workloads.build_loss(cfg)

    seed = cfg.get("experiment", {}).get("seed")
    _seed_all(seed)
    state = method.build(cfg)

    base_x = inputs["X"].detach()
    target = inputs.get("T")

    x_ref = base_x.clone().requires_grad_(True)
    if "W_full" in state:
        y_ref = torch.nn.functional.linear(x_ref, state["W_full"], state.get("bias"))
    elif "W" in state:
        y_ref = torch.nn.functional.linear(x_ref, state["W"], state.get("bias"))
    elif reference_method is not None:
        _seed_all(seed)
        ref_state = reference_method.build(cfg)
        y_ref = reference_method.forward(ref_state, x_ref)
    else:
        raise RuntimeError("reference method required for phase2")
    loss_ref = loss_fn(y_ref, target)
    loss_ref.backward()
    dX_ref = x_ref.grad.detach().clone()
    dX_ref_sum = None
    if dist_utils.rank() == 0:
        dX_ref_sum = float(dX_ref.sum().item())
        print(f"phase2_debug: dX_ref_sum={dX_ref_sum:.6e}", flush=True)
    if dist_utils.is_distributed():
        dX_ref_sum = dist_utils.broadcast_object(dX_ref_sum, src=0)

    if dist_utils.is_distributed():
        dist_utils.broadcast_tensor(dX_ref, src=0)

    x = base_x.clone().requires_grad_(True)
    loss_scale = 1.0
    if dist_utils.is_distributed():
        loss_scale = 1.0 / float(dist_utils.world_size())

    for _ in range(max(warmup, 0)):
        y = method.forward(state, x)
        loss = loss_fn(y, target) * loss_scale
        loss.backward()
        x.grad = None

    _reset_allreduce_samples(state)

    errors_max_abs: list[float] = []
    errors_mean_abs: list[float] = []
    errors_max_rel: list[float] = []
    last_output = None
    debug_grad_logged = False

    for _ in range(max(iters, 1)):
        with timers.time("forward"):
            last_output = method.forward(state, x)
        with timers.time("backward"):
            loss = loss_fn(last_output, target) * loss_scale
            loss.backward()
            if dist_utils.is_distributed() and x.grad is not None:
                if dist_utils.rank() == 0 and not debug_grad_logged:
                    grad_sum_pre = float(x.grad.sum().item())
                    print(
                        f"phase2_debug: grad_shape={tuple(x.grad.shape)} "
                        f"grad_sum_pre={grad_sum_pre:.6e}",
                        flush=True,
                    )
                dist_utils.allreduce_sum(x.grad, allow_autograd=False)
                if dist_utils.rank() == 0 and not debug_grad_logged:
                    grad_sum_post = float(x.grad.sum().item())
                    ratio = None
                    if dX_ref_sum not in (None, 0.0):
                        ratio = grad_sum_post / dX_ref_sum
                    max_abs_ref = float(dX_ref.abs().max().item())
                    max_abs_grad = float(x.grad.abs().max().item())
                    print(
                        f"phase2_debug: grad_sum_post={grad_sum_post:.6e} "
                        f"ratio={ratio if ratio is not None else 'n/a'} "
                        f"max_abs_ref={max_abs_ref:.6e} max_abs_grad={max_abs_grad:.6e}",
                        flush=True,
                    )
                    debug_grad_logged = True
        eps = float(phase.get("rel_eps", 1e-12))
        metrics = _compare_tensors(dX_ref, x.grad, eps=eps)
        errors_max_abs.append(metrics["max_abs_error"])
        errors_mean_abs.append(metrics["mean_abs_error"])
        errors_max_rel.append(metrics["max_rel_error"])
        x.grad = None

    tol_max_abs = float(phase.get("tol_max_abs", 0.0))
    tol_max_rel = float(phase.get("tol_max_rel", 0.0))
    passed = (max(errors_max_abs) <= tol_max_abs) and (max(errors_max_rel) <= tol_max_rel)

    results = {
        "phase": "phase2_backward_input",
        "grad_input_max_abs_error": max(errors_max_abs),
        "grad_input_mean_abs_error": sum(errors_mean_abs) / len(errors_mean_abs),
        "grad_input_max_rel_error": max(errors_max_rel),
        "passed": passed,
        "timings_ms": timers.summary(),
        "iterations": iters,
        "warmup_iters": warmup,
    }
    _attach_allreduce_timings(results, state)
    _attach_layout_fix_metrics(results, state)

    if logger is not None:
        log_metrics(logger, cfg, results)

    if dist_utils.rank() == 0:
        print(
            "phase2_backward_input:",
            f"passed={results['passed']}",
            f"max_abs_error={results['grad_input_max_abs_error']:.6e}",
            f"max_rel_error={results['grad_input_max_rel_error']:.6e}",
            f"mean_abs_error={results['grad_input_mean_abs_error']:.6e}",
            f"warmup_iters={warmup}",
            f"timed_iters={iters}",
            flush=True,
        )

    return PhaseResult(results=results, output=last_output)


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
    if phase_name == "phase2_backward_input":
        method_name = str(cfg.get("method", {}).get("name", ""))
        if method_name == "block_shock_2gpu":
            if logger is not None:
                results = {
                    "phase": "phase2_backward_input",
                    "skipped": True,
                    "skip_reason": "semi-structured sparse backward not supported",
                }
                log_metrics(logger, cfg, results)
            if dist_utils.rank() == 0:
                print(
                    "phase2_backward_input:",
                    "skipped=True",
                    "reason=semi-structured sparse backward not supported",
                    flush=True,
                )
            return PhaseResult(results={"skipped": True}, output=None)
        if reference_method is None:
            from .methods import dense_single as reference_method  # local import to avoid cycles
        return _run_phase2(cfg, method, inputs, reference_method, logger)

    raise ValueError(f"Unsupported phase: {phase_name}")
