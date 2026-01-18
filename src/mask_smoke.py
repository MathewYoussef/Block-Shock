### block_shock/src/mask_smoke.py
## Smoke test for 2:4 mask generation and validation.

from __future__ import annotations

import sys

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

from .sparsity import masks as mask_utils


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is required for mask smoke test")


def main() -> int:
    _require_torch()
    cfg = {
        "model": {"N": 8},
        "mask": {"pattern_a": "1100"},
    }

    masks = mask_utils.build_masks(cfg)
    m0 = masks["mask_a"]
    m1 = masks["mask_b"]

    mask_utils.validate_2of4(m0)
    mask_utils.validate_2of4(m1)
    mask_utils.validate_complementary(m0, m1)

    w = torch.randn_like(m0, dtype=torch.bfloat16)
    w_masked = w * m0.to(dtype=w.dtype)
    mask_utils.validate_masked_matrix(w_masked, m0)

    full_covered = torch.all(m0 | m1).item()
    overlap = torch.any(m0 & m1).item()
    sum_mask = m0.int() + m1.int()
    sum_all_ones = torch.all(sum_mask == 1).item()
    sum_unique = torch.unique(sum_mask).tolist()

    print(
        "mask_smoke:",
        f"m0_shape={tuple(m0.shape)}",
        f"m1_shape={tuple(m1.shape)}",
        f"full_covered={full_covered}",
        f"overlap={overlap}",
        f"sum_all_ones={sum_all_ones}",
        f"sum_unique={sum_unique}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
