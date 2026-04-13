import math
from functools import reduce

import torch
from torch import nn


def centers(start, stop, num, dtype=None, device=None):
    edges = torch.linspace(start, stop, num + 1, dtype=dtype, device=device)
    return (edges[:-1] + edges[1:]) / 2.0


def make_grid(h_pos, w_pos):
    grid = torch.stack(torch.meshgrid(h_pos, w_pos, indexing="ij"), dim=-1)
    # h, w, d = grid.shape
    # return grid.view(h * w, d)
    return grid.view(-1, 2)


def bounding_box(h, w):
    # Adjusted dimensions
    w_adj = w
    h_adj = h

    # Adjusted aspect ratio
    ar_adj = w_adj / h_adj

    # Determine bounding box based on the adjusted aspect ratio
    y_min, y_max, x_min, x_max = -1.0, 1.0, -1.0, 1.0
    if ar_adj > 1:
        y_min, y_max = -1 / ar_adj, 1 / ar_adj
    elif ar_adj < 1:
        x_min, x_max = -ar_adj, ar_adj

    return y_min, y_max, x_min, x_max


def make_axial_pos_2d(h: int, w: int, dtype=None, device=None):
    y_min, y_max, x_min, x_max = bounding_box(h, w)
    h_pos = centers(y_min, y_max, h, dtype=dtype, device=device)
    w_pos = centers(x_min, x_max, w, dtype=dtype, device=device)
    return make_grid(h_pos, w_pos)


def apply_rotary_emb(x: torch.Tensor, theta: torch.Tensor, conj=False):
    out_dtype = x.dtype
    dtype = reduce(torch.promote_types, (x.dtype, theta.dtype, torch.float32))
    d = theta.shape[-1]
    assert d * 2 <= x.shape[-1]
    x1, x2, x3 = x[..., :d], x[..., d : d * 2], x[..., d * 2 :]
    x1, x2, theta = x1.to(dtype), x2.to(dtype), theta.to(dtype)
    cos, sin = torch.cos(theta), torch.sin(theta)
    sin = -sin if conj else sin
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    y1, y2 = y1.to(out_dtype), y2.to(out_dtype)
    return torch.cat((y1, y2, x3), dim=-1)


class AxialRoPE3D(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        half_embedding: bool = True,
        yx_min_theta: float = 0.0,
        yx_max_theta: float = 40.0 * math.pi,
        t_min_theta: float = 0.0,
        t_max_theta: float = 10.0 * math.pi,
    ):
        super().__init__()

        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        yx_freqs = torch.stack(
            [
                torch.linspace(
                    math.log(yx_min_theta) if yx_min_theta > 0 else 0.0,
                    math.log(yx_max_theta),
                    n_heads * dim // 4 + 1,
                )[:-1].exp()
            ]
            * 2
        )
        self.yx_freqs = nn.Parameter(yx_freqs.view(2, dim // 4, n_heads).mT.contiguous())

        t_freqs = torch.linspace(
            math.log(t_min_theta) if t_min_theta > 0 else 0.0,
            math.log(t_max_theta),
            n_heads * dim // 4 + 1,
        )[:-1].exp()
        self.t_freqs = nn.Parameter(t_freqs.view(dim // 4, n_heads).mT.contiguous())

    def apply_emb(self, x, theta):
        return apply_rotary_emb(x, theta)

    def extra_repr(self):
        return f"dim={self.yx_freqs.shape[-2] * 4}, n_heads={self.yx_freqs.shape[-1]}"

    def forward(self, pos):
        theta_t = pos[..., None, 0:1] * self.t_freqs.to(pos.dtype)
        theta_h = pos[..., None, 1:2] * self.yx_freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 2:3] * self.yx_freqs[1].to(pos.dtype)
        return torch.cat((theta_t, theta_h, theta_w), dim=-1)


class AxialRoPE2D(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        half_embedding: bool = True,
        yx_min_theta: float = 0.0,
        yx_max_theta: float = 40.0 * math.pi,
    ):
        super().__init__()

        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        freqs = torch.stack(
            [
                torch.linspace(
                    math.log(yx_min_theta) if yx_min_theta > 0 else 0.0,
                    math.log(yx_max_theta),
                    n_heads * dim // 4 + 1,
                )[:-1].exp()
            ]
            * 2
        )
        self.freqs = nn.Parameter(freqs.view(2, dim // 4, n_heads).mT.contiguous())

    def apply_emb(self, x, theta):
        return apply_rotary_emb(x, theta)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-2] * 4}, n_heads={self.freqs.shape[-1]}"

    def forward(self, pos):
        theta_h = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        theta_w = pos[..., None, 1:2] * self.freqs[1].to(pos.dtype)
        return torch.cat((theta_h, theta_w), dim=-1)


class AxialRoPE1D(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        half_embedding: bool = True,
        theta: int = 1_000,
        learnable_freqs: bool = False,
    ):
        super().__init__()

        if half_embedding:
            assert dim % 2 == 0, "Half embedding is only supported for even dimensions"
            dim //= 2

        min_freq = 1 / theta
        max_freq = 1.0

        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.linspace(log_min, log_max, n_heads * dim // 2 + 1)[:-1].exp()
        freqs_ = freqs.view(1, dim // 2, n_heads).mT.contiguous()
        self.freqs = nn.Parameter(freqs_, requires_grad=learnable_freqs)

    def apply_emb(self, x, theta):
        return apply_rotary_emb(x, theta)

    def extra_repr(self):
        return f"dim={self.freqs.shape[-1] * 2}, n_heads={self.freqs.shape[-2]}"

    def forward(self, pos):
        theta = pos[..., None, 0:1] * self.freqs[0].to(pos.dtype)
        return theta
