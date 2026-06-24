from __future__ import annotations

import math
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .fmha_compat import fmha


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(
    features: torch.Tensor,
    coords: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos[coords].flatten(-2)
    sin = sin[coords].flatten(-2)
    return features * cos + rotate_half(features) * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, dim_model: int, freqs_for: str = "pixel", max_freq: int = 400) -> None:
        super().__init__()
        self.max_freq = max_freq
        self.freqs_for = freqs_for
        if freqs_for == "lang":
            inv_freq = 1.0 / (
                10000 ** (torch.arange(0, dim_model // 2, 2).float() / (dim_model // 2))
            )
        elif freqs_for == "pixel":
            inv_freq = torch.linspace(1.0, max_freq / 2, dim_model // 4) * pi
        else:
            raise ValueError(f"unsupported rotary frequency mode: {freqs_for}")
        self.register_buffer("inv_freq", inv_freq)
        self.max_coords = -1
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(coords.max().item())
        if self._cos_cached is None or seq_len > self.max_coords:
            self.max_coords = seq_len
            t = torch.arange(self.max_coords + 1, device=coords.device, dtype=self.inv_freq.dtype)
            if self.freqs_for == "pixel":
                t = t / self.max_freq
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.repeat_interleave(freqs, 2, -1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    def forward(self, features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        cos, sin = self._update_cos_sin_tables(coords)
        return apply_rotary_pos_emb(features, coords, cos, sin)


class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        rope: bool,
        rope_freqs: str,
    ) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.input_projection = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.output_projection = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.rope = RotaryEmbedding(n_embd // n_head, freqs_for=rope_freqs) if rope else None

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        q, k, v = self.input_projection(features).split(self.n_embd, dim=-1)
        k = k.view(1, features.shape[1], self.n_head, self.n_embd // self.n_head)
        q = q.view(1, features.shape[1], self.n_head, self.n_embd // self.n_head)
        if self.rope is not None:
            q = self.rope(q.transpose(1, 2), coords).transpose(1, 2)
            k = self.rope(k.transpose(1, 2), coords).transpose(1, 2)
        v = v.view(1, features.shape[1], self.n_head, self.n_embd // self.n_head)
        if not q.is_cuda:
            out = (
                torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=attn_bias.materialize((q.shape[1], k.shape[1])),
                    dropout_p=0,
                )
                .transpose(1, 2)
                .reshape(1, -1, self.n_embd)
            )
        else:
            out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(
                1,
                features.shape[1],
                self.n_embd,
            )
        return self.resid_dropout(self.output_projection(out))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        dropout: float,
        bias: bool,
        rope: bool,
        rope_freqs: str,
        resid_dropout: float = 0,
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd=n_embd, bias=bias)
        self.attn = SelfAttention(
            n_embd=n_embd,
            n_head=n_head,
            dropout=resid_dropout,
            bias=bias,
            rope=rope,
            rope_freqs=rope_freqs,
        )
        self.ln_2 = LayerNorm(n_embd=n_embd, bias=bias)
        self.mlp = MLP(n_embd=n_embd, dropout=dropout, bias=bias)

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        attention_output = self.attn(
            features=self.ln_1(features),
            coords=coords,
            attn_bias=attn_bias,
        )
        features = features + attention_output
        return features + self.mlp(self.ln_2(features))


class RoFormerEncoder(nn.Module):
    def __init__(
        self,
        n_attention_block: int,
        n_embd: int,
        n_head: int,
        dropout: float = 0.25,
        bias: bool = True,
        rope: bool = True,
        rope_freqs: str = "pixel",
        resid_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.attention_blocks = nn.ModuleList(
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                dropout=dropout,
                bias=bias,
                rope=rope,
                rope_freqs=rope_freqs,
                resid_dropout=resid_dropout,
            )
            for _ in range(n_attention_block)
        )

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        attn_bias: fmha.BlockDiagonalMask,
    ) -> torch.Tensor:
        for block in self.attention_blocks:
            features = block(features=features, coords=coords, attn_bias=attn_bias)
        return features
