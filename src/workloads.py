### block_shock/src/workloads.py
## Synthetic input and loss generation for benchmarks.

from __future__ import annotations

from typing import Any, Callable, Mapping
import math

try:  # Optional dependency during scaffolding
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
except Exception:  # pragma: no cover - allow import without torch
    torch = None
    F = None

#TODO: add more distributions or real data adapters as needed


def _get_dtype(name: str):
    if torch is None:
        raise RuntimeError("torch is required for workload generation")
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
        raise RuntimeError("torch is required for workload generation")
    workload = cfg.get("workload", {})
    device_name = workload.get("device")
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_generator(device, seed: int | None):
    if torch is None:
        raise RuntimeError("torch is required for workload generation")
    if seed is None:
        return None
    return torch.Generator(device=device).manual_seed(int(seed))


def build_inputs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("torch is required for workload generation")

    model = cfg.get("model", {})
    workload = cfg.get("workload", {})
    bsz = int(model.get("B", 0))
    dim = int(model.get("N", 0))
    if bsz <= 0 or dim <= 0:
        raise ValueError("model.B and model.N must be positive integers")

    dtype = _get_dtype(str(model.get("dtype", "float32")))
    device = _get_device(cfg)
    seed = workload.get("seed")
    if seed is None:
        seed = cfg.get("experiment", {}).get("seed")
    gen = _get_generator(device, seed)

    dist = str(workload.get("type", "random_normal")).lower()
    if dist in ("random_normal", "gaussian", "normal"):
        mean = float(workload.get("mean", 0.0))
        std = float(workload.get("std", 1.0))
        x = torch.randn((bsz, dim), device=device, dtype=dtype, generator=gen) * std + mean
    elif dist == "uniform":
        low = float(workload.get("low", -1.0))
        high = float(workload.get("high", 1.0))
        x = torch.rand((bsz, dim), device=device, dtype=dtype, generator=gen) * (high - low) + low
    elif dist == "activation_like":
        scale = float(workload.get("scale", 1.0))
        clip = float(workload.get("clip", 2.0))
        x = torch.randn((bsz, dim), device=device, dtype=dtype, generator=gen) * scale
        x = torch.clamp(x, -clip, clip)
    elif dist in ("transformer_mlp", "transformer_mlp_like"):
        if F is None:
            raise RuntimeError("torch.nn.functional is required for transformer_mlp workload")
        variant = str(workload.get("mlp_variant", "gelu")).lower()
        scale_in = float(workload.get("scale_in", 1.0))
        layernorm = bool(workload.get("layernorm", True))
        if variant == "swiglu":
            x1 = torch.randn((bsz, dim), device=device, dtype=dtype, generator=gen) * scale_in
            x2 = torch.randn((bsz, dim), device=device, dtype=dtype, generator=gen) * scale_in
            if layernorm:
                eps = float(workload.get("ln_eps", 1e-5))
                x1 = (x1 - x1.mean(dim=-1, keepdim=True)) / (x1.std(dim=-1, keepdim=True) + eps)
                x2 = (x2 - x2.mean(dim=-1, keepdim=True)) / (x2.std(dim=-1, keepdim=True) + eps)
            x = x1 * torch.sigmoid(x2)
        else:
            x = torch.randn((bsz, dim), device=device, dtype=dtype, generator=gen) * scale_in
            if layernorm:
                eps = float(workload.get("ln_eps", 1e-5))
                x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + eps)
            if variant == "gelu":
                x = F.gelu(x)
            elif variant in ("silu", "swish"):
                x = F.silu(x)
            elif variant == "relu":
                x = F.relu(x)
            else:
                raise ValueError(f"Unsupported mlp_variant: {variant}")
        scale_out = float(workload.get("scale_out", 1.0))
        x = x * scale_out
    elif dist in ("attention_like", "attention"):
        if F is None:
            raise RuntimeError("torch.nn.functional is required for attention_like workload")
        tokens = int(workload.get("tokens", bsz))
        batch = int(workload.get("attn_batch", 1))
        heads = int(workload.get("heads", 8))
        if batch * tokens != bsz:
            raise ValueError("attention_like requires model.B == attn_batch * tokens")
        if dim % heads != 0:
            raise ValueError("model.N must be divisible by heads for attention_like")
        head_dim = dim // heads
        q = torch.randn((batch, heads, tokens, head_dim), device=device, dtype=dtype, generator=gen)
        k = torch.randn((batch, heads, tokens, head_dim), device=device, dtype=dtype, generator=gen)
        v = torch.randn((batch, heads, tokens, head_dim), device=device, dtype=dtype, generator=gen)
        scale = float(workload.get("attn_scale", 1.0 / math.sqrt(head_dim)))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        x = out.reshape(batch * tokens, dim)
    elif dist in ("vision_conv", "conv_like"):
        if F is None:
            raise RuntimeError("torch.nn.functional is required for vision_conv workload")
        channels = int(workload.get("C", 3))
        height = int(workload.get("H", 32))
        width = int(workload.get("W", 32))
        out_channels = int(workload.get("out_channels", channels))
        kernel = int(workload.get("kernel_size", 3))
        stride = int(workload.get("stride", 1))
        padding = int(workload.get("padding", kernel // 2))
        img = torch.randn((bsz, channels, height, width), device=device, dtype=dtype, generator=gen)
        weight = torch.randn(
            (out_channels, channels, kernel, kernel),
            device=device,
            dtype=dtype,
            generator=gen,
        ) / math.sqrt(channels * kernel * kernel)
        x_conv = F.conv2d(img, weight, stride=stride, padding=padding)
        act = str(workload.get("activation", "relu")).lower()
        if act == "relu":
            x_conv = F.relu(x_conv)
        elif act == "gelu":
            x_conv = F.gelu(x_conv)
        elif act in ("silu", "swish"):
            x_conv = F.silu(x_conv)
        elif act in ("none", "linear"):
            pass
        else:
            raise ValueError(f"Unsupported activation: {act}")
        x = x_conv.flatten(1)
        if x.shape[1] != dim:
            raise ValueError("vision_conv output size does not match model.N; adjust C/H/W/out_channels")
    else:
        raise ValueError(f"Unsupported workload type: {dist}")

    target_type = workload.get("target")
    target = None
    if target_type:
        target_type = str(target_type).lower()
        if target_type == "zeros":
            target = torch.zeros_like(x)
        elif target_type == "random_normal":
            target = torch.randn_like(x, generator=gen)
        elif target_type == "uniform":
            low = float(workload.get("target_low", -1.0))
            high = float(workload.get("target_high", 1.0))
            target = torch.rand_like(x, generator=gen) * (high - low) + low
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

    return {"X": x, "T": target}


def build_loss(cfg: Mapping[str, Any]) -> Callable[[Any, Any | None], Any]:
    if torch is None:
        raise RuntimeError("torch is required for workload generation")
    workload = cfg.get("workload", {})
    loss_name = str(workload.get("loss", "sum")).lower()

    if loss_name == "sum":
        return lambda y, _t=None: y.sum()
    if loss_name in ("mse_zero", "mse"):
        return lambda y, _t=None: (y * y).mean()
    if loss_name == "mse_target":
        return lambda y, t=None: ((y - t) ** 2).mean()

    raise ValueError(f"Unsupported loss type: {loss_name}")
