import math
from functools import reduce
from typing import Literal

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from zipmo.rope import AxialRoPE1D, AxialRoPE2D, AxialRoPE3D


def zero_init(layer):
    nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rms_norm(x, scale, eps):
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return x * scale.to(x.dtype)


def linear_swiglu(x, weight, bias=None):
    x = x @ weight.mT
    if bias is not None:
        x = x + bias
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


def scale_for_cosine_sim(q, k, scale, eps):
    dtype = reduce(torch.promote_types, (q.dtype, k.dtype, scale.dtype, torch.float32))
    sum_sq_q = torch.sum(q.to(dtype) ** 2, dim=-1, keepdim=True)
    sum_sq_k = torch.sum(k.to(dtype) ** 2, dim=-1, keepdim=True)
    sqrt_scale = torch.sqrt(scale.to(dtype))
    scale_q = sqrt_scale * torch.rsqrt(sum_sq_q + eps)
    scale_k = sqrt_scale * torch.rsqrt(sum_sq_k + eps)
    return q * scale_q.to(q.dtype), k * scale_k.to(k.dtype)


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def extra_repr(self):
        return f"shape={tuple(self.scale.shape)}, eps={self.eps}"

    def forward(self, x):
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    def __init__(self, features, cond_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def forward(self, x, cond):
        return rms_norm(x, self.linear(cond)[:, None, :] + 1, self.eps)


class LinearSwiGLU(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, input):
        return linear_swiglu(input, self.weight, self.bias)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, cond_features=None):
        super().__init__()
        self.norm = AdaRMSNorm(d_model, cond_features) if cond_features is not None else RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x, cond_norm=None, **kwargs):
        skip = x
        x = self.norm(x, cond_norm) if cond_norm is not None else self.norm(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        return x + skip


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond_norm: int | None = None,
        d_head: int = 64,
        rope_mode: Literal["1d", "2d", "3d_10", "3d_40"] = "3d_40",
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.rope_mode = rope_mode
        self.norm = AdaRMSNorm(d_model, d_cond_norm) if d_cond_norm is not None else RMSNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, d_model * 3, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.eps = 1e-6
        if rope_mode == "3d_40":
            self.pos_emb = AxialRoPE3D(d_head, self.n_heads)
        elif rope_mode == "3d_10":
            self.pos_emb = AxialRoPE3D(d_head, self.n_heads, yx_max_theta=10.0 * 3.14)
        elif rope_mode == "2d":
            self.pos_emb = AxialRoPE2D(d_head, self.n_heads)
        elif rope_mode == "1d":
            self.pos_emb = AxialRoPE1D(d_head, self.n_heads)
        else:
            raise ValueError(f"Unknown rope_mode for SelfAttentionBlock: {rope_mode}")

    def forward(self, x, pos, cond_norm=None, **kwargs):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b l (t nh e) -> t b nh l e", t=3, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], torch.tensor(self.eps, device=x.device))

        theta = self.pos_emb(pos.to(qkv.dtype)).movedim(-2, -3)
        q = self.pos_emb.apply_emb(q, theta)
        k = self.pos_emb.apply_emb(k, theta)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = einops.rearrange(x, "b nh l e -> b l (nh e)")
        x = self.out_proj(x)
        return x + skip


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int,
        d_cond_norm: int | None = None,
        d_head: int = 64,
        rope_mode: Literal["none", "1d", "2d", "3d_10", "3d_40"] = "3d_40",
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = d_model // d_head
        self.rope_mode = rope_mode
        self.norm = AdaRMSNorm(d_model, d_cond_norm) if d_cond_norm is not None else RMSNorm(d_model)
        self.norm_cross = RMSNorm(d_cross)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_cross, d_model * 2, bias=False)
        self.scale = nn.Parameter(torch.full([self.n_heads], 10.0))
        self.out_proj = zero_init(nn.Linear(d_model, d_model, bias=False))
        self.eps = 1e-6
        if rope_mode == "3d_40":
            self.pos_emb = AxialRoPE3D(d_head, self.n_heads)
        elif rope_mode == "3d_10":
            self.pos_emb = AxialRoPE3D(d_head, self.n_heads, yx_max_theta=10.0 * 3.14)
        elif rope_mode == "2d":
            self.pos_emb = AxialRoPE2D(d_head, self.n_heads)
        elif rope_mode == "1d":
            self.pos_emb = AxialRoPE1D(d_head, self.n_heads)
        elif rope_mode == "none":
            self.pos_emb = None
        else:
            raise ValueError(f"Unknown rope_mode for CrossAttentionBlock: {rope_mode}")

    def forward(self, x, pos, x_cross, pos_cross, cond_norm=None, **kwargs):
        skip = x
        if cond_norm is not None:
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x_cross = self.norm_cross(x_cross)

        q = self.q_proj(x)
        kv = self.kv_proj(x_cross)
        q = einops.rearrange(q, "b l (nh e) -> b nh l e", e=self.d_head)
        k, v = einops.rearrange(kv, "b l (t nh e) -> t b nh l e", t=2, e=self.d_head)
        q, k = scale_for_cosine_sim(q, k, self.scale[:, None, None], torch.tensor(self.eps, device=x.device))

        if self.pos_emb is not None:
            theta = self.pos_emb(pos.to(q.dtype)).movedim(-2, -3)
            theta_cross = self.pos_emb(pos_cross.to(q.dtype)).movedim(-2, -3)
            q = self.pos_emb.apply_emb(q, theta)
            k = self.pos_emb.apply_emb(k, theta_cross)

        x = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        x = einops.rearrange(x, "b nh l e -> b l (nh e)")
        x = self.out_proj(x)
        return x + skip


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cross: int | None = None,
        d_cond_norm: int | None = None,
        d_head=64,
        ff_expand=3,
        self_rope_mode: Literal["1d", "2d", "3d_10", "3d_40"] = "3d_40",
        cross_rope_mode: Literal["none", "1d", "2d", "3d_10", "3d_40"] | None = None,
        use_ca: bool = True,
    ):
        super().__init__()
        if cross_rope_mode is None:
            cross_rope_mode = self_rope_mode

        d_ff = d_model * ff_expand
        self.use_ca = use_ca
        self.self_attn = SelfAttentionBlock(
            d_model,
            d_cond_norm=d_cond_norm,
            d_head=d_head,
            rope_mode=self_rope_mode,
        )
        if self.use_ca:
            if d_cross is None:
                raise ValueError("d_cross must be provided when use_ca=True.")
            self.cross_attn = CrossAttentionBlock(
                d_model,
                d_cross=d_cross,
                d_cond_norm=d_cond_norm,
                d_head=d_head,
                rope_mode=cross_rope_mode,
            )
        else:
            self.cross_attn = None
        self.ff = FeedForwardBlock(d_model, d_ff, cond_features=d_cond_norm)

    def forward(self, x, pos, **kwargs):
        x = self.self_attn(x, pos, **kwargs)
        if self.cross_attn is not None:
            x = self.cross_attn(x, pos, **kwargs)
        x = self.ff(x, **kwargs)
        return x


class Level(nn.ModuleList):
    def forward(self, x, *args, **kwargs):
        for layer in self:
            x = layer(x, *args, **kwargs)
        return x


class CondTokenConcatMerge(nn.Module):
    def __init__(self, in_features: int, out_features: int, extra_token_features: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.extra_proj = nn.Linear(extra_token_features, out_features, bias=False)

    def forward(self, x, pos, extra_tokens, extra_pos, **kwargs):
        return torch.cat((self.proj(x), self.extra_proj(extra_tokens)), dim=1), torch.cat((pos, extra_pos), dim=1)


class CondTokenConcatSplit(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_extra_tokens: int):
        super().__init__()
        self.num_extra_tokens = num_extra_tokens
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x[:, : -self.num_extra_tokens])


class SimpleProjIn(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, pos, **kwargs):
        return self.proj(x), pos


class SimpleProj(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, *args, **kwargs):
        return self.proj(x)


class InputMLP(nn.Module):
    coeffs: torch.Tensor

    def __init__(
        self,
        in_features: int,
        dim: int,
        depth: int = 2,
        random_fourier: bool = False,
        random_fourier_paper: bool = False,
        theta=40.0 * math.pi,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            *[FeedForwardBlock(dim, 3 * dim) for _ in range(depth)],
        )

        if random_fourier:
            assert not random_fourier_paper, "Cannot use both random_fourier and random_fourier_paper"
            std = theta / 1.96  # 95% percentile standard normal distribution
            coeffs = torch.randn(dim // (2 * in_features)) * std
        elif random_fourier_paper:
            std = theta / 2.447  # 95% percentile Rayleigh distribution
            coeffs = torch.randn(dim // 2, in_features) * std
        else:
            coeffs = torch.linspace(0, math.log(theta), dim // (2 * in_features) + 1)[:-1].exp()
        self.register_buffer("coeffs", coeffs)

        self.random_fourier_paper = random_fourier_paper
        self.n_pad = dim % (2 * in_features)  # If dim is not divisible by 2*in_features, we have to pad before MLP

    def forward(
        self,
        tracks: Float[torch.Tensor, "... d_tracks"],
    ):
        coeffs = self.coeffs.to(tracks)

        if self.random_fourier_paper:  # random linear combinations of in_features, then apply sin and cos
            tracks_fourier: Float[torch.Tensor, "... n_coeff"] = tracks @ coeffs.mT
            n_flatten_dims = 2
        else:
            coeffs = coeffs.view(*([1] * len(tracks.shape)), -1)
            tracks_fourier: Float[torch.Tensor, "... d_tracks n_coeff"] = tracks[..., None] * coeffs
            # we have an extra dim here
            n_flatten_dims = 3

        stacked = torch.stack(
            [torch.sin(tracks_fourier), torch.cos(tracks_fourier)], dim=-2
        )  # (b, n, t, (d_tracks), 2 , n_coeff)

        # flatten trailing dims into a single feature dim
        # tracks_fourier = stacked.reshape(*stacked.shape[:-n_flatten_dims], -1)
        # below works better with 0 dims, e.g. not conditioning tracks
        tracks_fourier = torch.flatten(stacked, start_dim=-n_flatten_dims)
        pad_tensor = torch.zeros((*tracks_fourier.shape[:-1], self.n_pad), device=tracks_fourier.device)
        tracks_fourier = torch.cat([tracks_fourier, pad_tensor], dim=-1)
        out = self.mlp(tracks_fourier)

        return out


class OutputMLP(nn.Sequential):
    def __init__(self, width: int, d_out: int, depth: int = 2, **kwargs):
        super().__init__(
            RMSNorm(width),
            *[l for _ in range(depth) for l in [nn.Linear(width, width), nn.SiLU()]],
            nn.Linear(width, d_out),
        )
