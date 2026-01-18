### block_shock/src/methods/dense_single.py
## Dense single-GPU baseline.

from __future__ import annotations

from typing import Any, Mapping

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    F = None

from ..utils import nudge_zeros


def _get_dtype(name: str):
    if torch is None:
        raise RuntimeError("torch is required for dense_single")
    key = name.lower()
    if key in ("bf16", "bfloat16"):
        return torch.bfloat16
    if key in ("fp16", "float16", "half"):
        return torch.float16
    if key in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _get_device(cfg: Mapping[str, Any]):
    if torch is None:
        raise RuntimeError("torch is required for dense_single")
    method = cfg.get("method", {})
    workload = cfg.get("workload", {})
    hardware = cfg.get("hardware", {})
    for source in (method, workload, hardware):
        device = source.get("device")
        if device:
            return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for dense_single")
    model = cfg.get("model", {})
    n = int(model.get("N", 0))
    if n <= 0:
        raise ValueError("model.N must be set for dense_single")
    dtype = _get_dtype(str(model.get("dtype", "float32")))
    device = _get_device(cfg)

    requires_grad = bool(cfg.get("phase", {}).get("train_step", False))
    weight = torch.randn((n, n), device=device, dtype=dtype, requires_grad=requires_grad)
    weight = nudge_zeros(weight)

    bias = None
    if cfg.get("method", {}).get("bias", False):
        bias = torch.zeros((n,), device=device, dtype=dtype, requires_grad=requires_grad)

    optimizer = None
    lr = cfg.get("method", {}).get("lr")
    if requires_grad and lr is not None:
        optimizer = torch.optim.SGD([weight] + ([bias] if bias is not None else []), lr=float(lr))

    return {"W": weight, "bias": bias, "optimizer": optimizer}


def forward(state: Mapping[str, Any], x):
    if F is None:
        raise RuntimeError("torch.nn.functional is required for dense_single")
    return F.linear(x, state["W"], state.get("bias"))


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
