### block_shock/src/semistructured_smoke.py
## Smoke test: compress masked weight and compare sparse vs dense matmul.

from __future__ import annotations

import sys

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    F = None

from .sparsity import masks as mask_utils
from .sparsity import semistructured as ss


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for semistructured smoke test")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for semistructured smoke test")


def main() -> int:
    _require_torch()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    n = 4096
    b = 64

    cfg = {
        "model": {"N": n},
        "mask": {"pattern_a": "1100"},
    }

    masks = mask_utils.build_masks(cfg)
    m0 = masks["mask_a"].to(device=device)
    mask_utils.validate_2of4(m0)

    w = torch.randn((n, n), device=device, dtype=dtype)
    w_masked = w * m0.to(dtype=dtype)
    mask_utils.validate_masked_matrix(w_masked, m0)

    w_sparse = ss.compress(w_masked)
    x = torch.randn((b, n), device=device, dtype=dtype)

    y_dense = F.linear(x, w_masked, None)
    try:
        y_sparse = F.linear(x, w_sparse, None)
        op_used = "F.linear"
    except Exception:
        y_sparse = torch.mm(x, w_sparse.t())
        op_used = "torch.mm"

    diff = (y_dense - y_sparse).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    denom = y_dense.abs()
    max_rel = float((diff / (denom + 1e-12)).max().item())

    print(
        "semistructured_smoke:",
        f"n={n}",
        f"b={b}",
        f"dtype={dtype}",
        f"op={op_used}",
        f"max_abs_error={max_abs:.6e}",
        f"mean_abs_error={mean_abs:.6e}",
        f"max_rel_error={max_rel:.6e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
