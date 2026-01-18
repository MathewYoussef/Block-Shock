### block_shock/src/sparsity/semistructured.py
## Semi-structured compression/decompression and guardrails.

from __future__ import annotations

from typing import Iterable

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None

from . import masks as mask_utils

#TODO: guard supported ops to avoid silent dense fallback

SUPPORTED_OPS = {"mm", "addmm", "linear"}


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for semi-structured compression")


def _to_sparse_semi_structured(weights):
    if hasattr(weights, "to_sparse_semi_structured"):
        return weights.to_sparse_semi_structured()
    if hasattr(torch, "to_sparse_semi_structured"):
        return torch.to_sparse_semi_structured(weights)
    if hasattr(torch, "sparse") and hasattr(torch.sparse, "to_sparse_semi_structured"):
        return torch.sparse.to_sparse_semi_structured(weights)
    raise RuntimeError("to_sparse_semi_structured is not available in this torch build")


def _validate_constraints(weights) -> None:
    if weights.ndim != 2:
        raise ValueError("semi-structured requires 2D weights")
    if not weights.is_cuda:
        raise ValueError("semi-structured requires CUDA tensors")
    if weights.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("semi-structured requires fp16 or bf16 weights")
    if weights.shape[0] % 64 != 0 or weights.shape[1] % 64 != 0:
        raise ValueError("semi-structured requires both dimensions divisible by 64")


def guard_supported_op(op_name: str) -> None:
    if op_name not in SUPPORTED_OPS:
        raise ValueError(f"unsupported op for semi-structured path: {op_name}")


def validate_2of4_weights(weights, atol: float = 0.0) -> None:
    _require_torch()
    mask = weights.abs() > atol
    mask_utils.validate_2of4(mask)


def compress(_weights):
    _require_torch()
    _validate_constraints(_weights)
    return _to_sparse_semi_structured(_weights)


def decompress(_weights):
    _require_torch()
    if hasattr(_weights, "to_dense"):
        return _weights.to_dense()
    raise RuntimeError("weights do not support to_dense() for decompression")
