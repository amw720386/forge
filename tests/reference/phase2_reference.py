#!/usr/bin/env python3
"""PyTorch reference for Phase 2 primitives + tiny GQA decoder attention.

Writes binary fixtures consumed by Rust tests under runtime/tests/fixtures/.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runtime" / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    t = tensor.detach().cpu().contiguous().float()
    path.write_bytes(t.numpy().tobytes())


def write_meta(name: str, meta: dict) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [T, H]
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    # x: [T, n_heads, D] — HF/Qwen2 half-rotation RoPE
    _t, _n_heads, d = x.shape
    half = d // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / d))
    freqs = torch.outer(positions.float(), inv_freq)  # [T, half]
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1)[:, None, :]
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1)[:, None, :]
    return x * cos + rotate_half(x) * sin


def gen_primitives() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 8)
    w = torch.ones(8)
    y = rmsnorm(x, w, 1e-6)
    write_f32(OUT / "rmsnorm_x.bin", x)
    write_f32(OUT / "rmsnorm_w.bin", w)
    write_f32(OUT / "rmsnorm_y.bin", y)
    write_meta("rmsnorm", {"rows": 3, "dim": 8, "eps": 1e-6})

    logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]], dtype=torch.float32)
    sm = F.softmax(logits, dim=-1)
    write_f32(OUT / "softmax_in.bin", logits)
    write_f32(OUT / "softmax_out.bin", sm)
    write_meta("softmax", {"rows": 2, "cols": 3})

    # Causal mask row-wise
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    mask = torch.triu(torch.ones(2, 2), diagonal=1).bool()
    scores_c = scores.masked_fill(mask, float("-inf"))
    sm_c = F.softmax(scores_c, dim=-1)
    write_f32(OUT / "softmax_causal_in.bin", scores)
    write_f32(OUT / "softmax_causal_out.bin", sm_c)

    x = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)
    write_f32(OUT / "silu_x.bin", x)
    write_f32(OUT / "silu_y.bin", silu(x))


def gen_rope() -> None:
    torch.manual_seed(1)
    t, n_heads, d = 4, 2, 8
    x = torch.randn(t, n_heads, d)
    positions = torch.arange(t)
    y = apply_rope(x.clone(), positions, 10000.0)
    write_f32(OUT / "rope_x.bin", x.reshape(-1))
    write_f32(OUT / "rope_y.bin", y.reshape(-1))
    write_meta("rope", {"seq": t, "n_heads": n_heads, "head_dim": d, "theta": 10000.0})


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_heads: int,
    n_kv: int,
    causal: bool,
) -> torch.Tensor:
    """q/k/v: [T, H] projected; returns [T, n_heads*D]."""
    t, q_dim = q.shape
    d = q_dim // n_heads
    kv_dim = k.shape[1]
    assert kv_dim // n_kv == d
    q = q.view(t, n_heads, d)
    k = k.view(t, n_kv, d)
    v = v.view(t, n_kv, d)
    reps = n_heads // n_kv
    k = k.repeat_interleave(reps, dim=1)
    v = v.repeat_interleave(reps, dim=1)
    scale = 1.0 / math.sqrt(d)
    # [n_heads, T, T]
    scores = torch.einsum("thd,Thd->htT", q, k) * scale
    if causal:
        mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))
    probs = F.softmax(scores, dim=-1)
    ctx = torch.einsum("htT,Thd->thd", probs, v)
    return ctx.reshape(t, n_heads * d)


def gen_attention() -> None:
    torch.manual_seed(2)
    t, n_heads, n_kv, d = 5, 4, 2, 8
    q = torch.randn(t, n_heads * d)
    k = torch.randn(t, n_kv * d)
    v = torch.randn(t, n_kv * d)
    # Apply rope to q/k as [T, heads, D]
    pos = torch.arange(t)
    q_r = apply_rope(q.view(t, n_heads, d).clone(), pos, 10000.0).reshape(t, -1)
    k_r = apply_rope(k.view(t, n_kv, d).clone(), pos, 10000.0).reshape(t, -1)
    ctx = attention_reference(q_r, k_r, v, n_heads, n_kv, causal=True)
    write_f32(OUT / "attn_q.bin", q)
    write_f32(OUT / "attn_k.bin", k)
    write_f32(OUT / "attn_v.bin", v)
    write_f32(OUT / "attn_ctx.bin", ctx)
    write_meta(
        "attn",
        {
            "seq": t,
            "n_heads": n_heads,
            "n_kv": n_kv,
            "head_dim": d,
            "theta": 10000.0,
        },
    )


def main() -> None:
    gen_primitives()
    gen_rope()
    gen_attention()
    print(f"Wrote fixtures to {OUT}")


if __name__ == "__main__":
    main()
