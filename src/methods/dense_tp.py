### block_shock/src/methods/dense_tp.py
## Dense 2-GPU tensor parallel baseline (row-parallel preferred).

from __future__ import annotations

import os
import time
from typing import Any, Mapping

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    F = None

from .. import distributed as dist_utils
from ..utils import nudge_zeros


def _get_dtype(name: str):
    if torch is None:
        raise RuntimeError("torch is required for dense_tp")
    key = name.lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _get_device() -> "torch.device":
    if torch is None:
        raise RuntimeError("torch is required for dense_tp")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def _shard_range(n: int, world_size: int, rank: int) -> tuple[int, int]:
    if n % world_size != 0:
        raise ValueError("model.N must be divisible by world_size for row-parallel TP")
    shard = n // world_size
    start = rank * shard
    end = start + shard
    return start, end


def build(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for dense_tp")
    model = cfg.get("model", {})
    n = int(model.get("N", 0))
    if n <= 0:
        raise ValueError("model.N must be set for dense_tp")
    dtype = _get_dtype(str(model.get("dtype", "float32")))
    device = _get_device()

    world_size = dist_utils.world_size()
    rank = dist_utils.rank()
    start, end = _shard_range(n, world_size, rank)

    requires_grad = bool(cfg.get("phase", {}).get("train_step", False))
    full_weight = torch.randn((n, n), device=device, dtype=dtype, requires_grad=requires_grad)
    full_weight = nudge_zeros(full_weight)
    if dist_utils.is_distributed():
        dist_utils.broadcast_tensor(full_weight, src=0)
    weight_shard = full_weight[:, start:end].contiguous()

    bias = None
    if cfg.get("method", {}).get("bias", False):
        bias = torch.zeros((n,), device=device, dtype=dtype, requires_grad=requires_grad)
        if dist_utils.is_distributed():
            dist_utils.broadcast_tensor(bias, src=0)

    optimizer = None
    lr = cfg.get("method", {}).get("lr")
    params = [weight_shard] + ([bias] if bias is not None else [])
    if requires_grad and lr is not None:
        optimizer = torch.optim.SGD(params, lr=float(lr))

    phase_cfg = cfg.get("phase", {})
    record_allreduce = bool(phase_cfg.get("record_timings", False))
    layout_fix_mode = str(phase_cfg.get("sync_mode", "none"))

    debug = bool(cfg.get("method", {}).get("debug", False) or os.environ.get("DENSE_TP_DEBUG") == "1")
    if debug:
        weight_sum = float(full_weight.sum().item())
        print(
            f"dense_tp build rank={rank} world_size={world_size} shard=[{start}:{end}] "
            f"weight_sum={weight_sum:.6e}",
            flush=True,
        )

    return {
        "W_full": full_weight,
        "W_shard": weight_shard,
        "bias": bias,
        "start": start,
        "end": end,
        "optimizer": optimizer,
        "record_allreduce": record_allreduce,
        "layout_fix_mode": layout_fix_mode,
        "allreduce_event_pairs": [],
        "allreduce_samples_ms": [],
        "layout_fix_event_pairs": [],
        "layout_fix_samples_ms": [],
        "layout_fix_did_copy": [],
        "layout_fix_bytes": [],
        "debug": debug,
        "debug_printed": False,
    }


def forward(state: Mapping[str, Any], x):
    if F is None:
        raise RuntimeError("torch.nn.functional is required for dense_tp")
    start, end = int(state["start"]), int(state["end"])
    x_shard = x[:, start:end].contiguous()
    y_partial = F.linear(x_shard, state["W_shard"], None)
    y_ready, meta, event_pair, duration_ms = dist_utils.collective_prep(
        y_partial, timing_mode=str(state.get("layout_fix_mode", "none"))
    )
    state["layout_fix_did_copy"].append(bool(meta["did_copy"]))
    state["layout_fix_bytes"].append(int(meta["bytes"]))
    if event_pair is not None:
        state["layout_fix_event_pairs"].append(event_pair)
    elif duration_ms is not None:
        state["layout_fix_samples_ms"].append(float(duration_ms))
    if state.get("record_allreduce"):
        if torch is not None and torch.cuda.is_available():
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record(torch.cuda.current_stream())
            dist_utils.allreduce_sum(y_ready)
            end_evt.record(torch.cuda.current_stream())
            state["allreduce_event_pairs"].append((start_evt, end_evt))
        else:
            t0 = time.perf_counter()
            dist_utils.allreduce_sum(y_ready)
            state["allreduce_samples_ms"].append((time.perf_counter() - t0) * 1e3)
    else:
        dist_utils.allreduce_sum(y_ready)
    y = y_ready
    bias = state.get("bias")
    if bias is not None:
        y = y + bias
    if state.get("debug") and not state.get("debug_printed"):
        state["debug_printed"] = True
        x_sum = float(x_shard.sum().item())
        y_sum = float(y.sum().item())
        print(
            f"dense_tp forward rank={dist_utils.rank()} x_shard_sum={x_sum:.6e} y_sum={y_sum:.6e}",
            flush=True,
        )
    return y


def backward(_state: Mapping[str, Any], loss) -> None:
    loss.backward()


def step(state: Mapping[str, Any]) -> None:
    optimizer = state.get("optimizer")
    if optimizer is None:
        return
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def teardown(_state: Mapping[str, Any]) -> None:
    #TODO: cleanup if needed
    pass
