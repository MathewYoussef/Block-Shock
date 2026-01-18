### block_shock/src/sparsity/masks.py
## Mask generation and 2:4 validation helpers.

from __future__ import annotations

from typing import Any, Iterable, Mapping
import re

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None

#TODO: later test:
#TODO: - random per-row patterns
#TODO: - magnitude-based "choose best 2-of-4" per block
#TODO: - dynamic masks


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for mask generation and validation")


def _parse_pattern(pattern: Any) -> list[int]:
    if isinstance(pattern, str):
        digits = [ch for ch in pattern if ch in ("0", "1")]
        values = [int(ch) for ch in digits]
    elif isinstance(pattern, Iterable):
        values = [int(v) for v in pattern]
    else:
        raise ValueError("pattern must be a string or iterable of 0/1 values")
    if len(values) != 4:
        raise ValueError(f"pattern must be length 4, got {len(values)}")
    if any(v not in (0, 1) for v in values):
        raise ValueError("pattern must contain only 0/1 values")
    if sum(values) != 2:
        raise ValueError("pattern must have exactly two 1s (2-of-4)")
    return values


def get_pattern_from_cfg(mask_cfg: Mapping[str, Any], rank: int) -> list[int]:
    pattern_a = mask_cfg.get("pattern_a") or mask_cfg.get("pattern")
    pattern_b = mask_cfg.get("pattern_b")
    if pattern_a is None:
        name = str(mask_cfg.get("name", ""))
        matches = re.findall(r"[01]{4}", name)
        if matches:
            pattern_a = matches[0]
            if pattern_b is None and len(matches) > 1:
                pattern_b = matches[1]
    if pattern_a is None:
        raise ValueError("mask.pattern_a or mask.pattern must be set")
    parsed_a = _parse_pattern(pattern_a)
    if pattern_b is None:
        parsed_b = [1 - v for v in parsed_a]
    else:
        parsed_b = _parse_pattern(pattern_b)
    return parsed_a if rank == 0 else parsed_b


def apply_pattern_inplace(weight: "torch.Tensor", pattern: list[int]) -> None:
    _require_torch()
    for idx, keep in enumerate(pattern):
        if keep == 0:
            weight[:, idx::4] = 0


def _mask_from_pattern(pattern: Any, rows: int, cols: int, device=None) -> "torch.Tensor":
    _require_torch()
    if cols % 4 != 0:
        raise ValueError("mask last dimension must be divisible by 4")
    pattern_vals = _parse_pattern(pattern)
    base = torch.tensor(pattern_vals, dtype=torch.bool, device=device)
    tiled = base.repeat(cols // 4)
    mask = tiled.unsqueeze(0).repeat(rows, 1)
    return mask


def generate_complement(mask: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    return ~mask.bool()


def validate_2of4(mask: "torch.Tensor", chunk_blocks: int = 1_000_000) -> None:
    _require_torch()
    if mask.ndim < 1:
        raise ValueError("mask must have at least one dimension")
    if mask.shape[-1] % 4 != 0:
        raise ValueError("mask last dimension must be divisible by 4")
    mask_bool = mask.bool()
    view = mask_bool.reshape(-1, 4)
    total_blocks = view.shape[0]
    min_count = 4
    max_count = 0
    for start in range(0, total_blocks, chunk_blocks):
        chunk = view[start : start + chunk_blocks]
        counts = chunk.sum(dim=-1)
        if not torch.all(counts == 2):
            min_count = int(counts.min().item())
            max_count = int(counts.max().item())
            raise ValueError(f"mask violates 2-of-4 rule (min={min_count}, max={max_count})")
        min_count = min(min_count, int(counts.min().item()))
        max_count = max(max_count, int(counts.max().item()))


def validate_complementary(mask_a: "torch.Tensor", mask_b: "torch.Tensor") -> None:
    _require_torch()
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask shapes must match for complement validation")
    a = mask_a.bool()
    b = mask_b.bool()
    overlap = torch.any(a & b)
    if overlap:
        raise ValueError("mask complement check failed (overlap detected)")
    covered = torch.all(a | b)
    if not covered:
        raise ValueError("mask complement check failed (not fully covered)")


def validate_masked_matrix(matrix: "torch.Tensor", mask: "torch.Tensor", atol: float = 0.0) -> None:
    _require_torch()
    if matrix.shape != mask.shape:
        raise ValueError("matrix and mask must have the same shape")
    validate_2of4(mask)
    masked_values = matrix[~mask.bool()]
    if masked_values.numel() == 0:
        return
    max_abs = float(masked_values.abs().max().item())
    if max_abs > atol:
        raise ValueError(f"masked matrix has nonzeros above atol={atol} (max_abs={max_abs:.6e})")


def build_masks(cfg: Mapping[str, Any]) -> dict[str, "torch.Tensor"]:
    _require_torch()
    mask_cfg = cfg.get("mask", {})
    model_cfg = cfg.get("model", {})
    n = int(model_cfg.get("N", 0))
    if n <= 0:
        raise ValueError("model.N must be set to build masks")

    mode = str(mask_cfg.get("mode", "pattern")).lower()
    if mode not in ("pattern", "complement"):
        raise NotImplementedError(f"mask mode '{mode}' not implemented yet")

    pattern_a = mask_cfg.get("pattern_a") or mask_cfg.get("pattern")
    pattern_b = mask_cfg.get("pattern_b")
    if pattern_a is None:
        name = str(mask_cfg.get("name", ""))
        matches = re.findall(r"[01]{4}", name)
        if matches:
            pattern_a = matches[0]
            if pattern_b is None and len(matches) > 1:
                pattern_b = matches[1]
    if pattern_a is None:
        raise ValueError("mask.pattern_a or mask.pattern must be set")

    mask_a = _mask_from_pattern(pattern_a, rows=n, cols=n)

    if pattern_b is None:
        mask_b = generate_complement(mask_a)
    else:
        mask_b = _mask_from_pattern(pattern_b, rows=n, cols=n)

    validate_2of4(mask_a)
    validate_2of4(mask_b)
    validate_complementary(mask_a, mask_b)

    return {"mask_a": mask_a, "mask_b": mask_b}
