### block_shock/src/workload_gen_repetition_smoke.py
## Smoke test: generate each workload twice with same seed and compare X.

from __future__ import annotations

import os

try:
    import torch  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency during scaffolding
    torch = None
    _torch_error = exc
else:
    _torch_error = None

from .workloads import build_inputs


def _maybe_enable_determinism() -> None:
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_case(name: str, cfg: dict) -> bool:
    out1 = build_inputs(cfg)["X"]
    out2 = build_inputs(cfg)["X"]
    same = torch.equal(out1, out2)
    device = out1.device
    print(f"{name}: same={same} shape={tuple(out1.shape)} dtype={out1.dtype} device={device}")
    return same


def main() -> None:
    if torch is None:
        print(f"torch import failed: {_torch_error}")
        return

    _maybe_enable_determinism()

    device = os.environ.get("WORKLOAD_DEVICE")
    dtype = os.environ.get("WORKLOAD_DTYPE", "float32")
    base_model = {"N": 4096, "B": 64, "dtype": dtype}
    if device:
        base_workload = {"device": device}
    else:
        base_workload = {}

    cases = [
        ("random_normal", {"workload": {"type": "random_normal", "seed": 1234}}),
        ("uniform", {"workload": {"type": "uniform", "seed": 1234}}),
        ("activation_like", {"workload": {"type": "activation_like", "seed": 1234}}),
        (
            "transformer_mlp",
            {
                "workload": {
                    "type": "transformer_mlp",
                    "seed": 1234,
                    "mlp_variant": "gelu",
                    "layernorm": True,
                }
            },
        ),
        (
            "attention_like",
            {
                "workload": {
                    "type": "attention_like",
                    "seed": 1234,
                    "attn_batch": 1,
                    "tokens": 64,
                    "heads": 8,
                }
            },
        ),
        (
            "vision_conv",
            {
                "workload": {
                    "type": "vision_conv",
                    "seed": 1234,
                    "C": 3,
                    "H": 32,
                    "W": 32,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "activation": "relu",
                }
            },
        ),
    ]

    all_ok = True
    for name, overrides in cases:
        cfg = {"model": dict(base_model), "workload": dict(base_workload)}
        cfg["workload"].update(overrides["workload"])
        ok = _run_case(name, cfg)
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit("workload repetition smoke test failed")


if __name__ == "__main__":
    main()
