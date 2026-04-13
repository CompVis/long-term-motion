# Source: https://github.com/cloneofsimo/minDinoV2

import contextlib
import math
from functools import partial
from typing import Literal

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float


def better_resize(imgs: torch.Tensor, image_size: int) -> torch.Tensor:
    ss = imgs.shape
    assert ss[-3] == 3  # channels first

    H, W = ss[-2:]

    # fast exit for already correct size
    if H == image_size and W == image_size:
        return imgs

    if len(ss) == 3:
        imgs = imgs.unsqueeze(0)

    side = min(H, W)
    int_factor: int = side // image_size

    if int_factor > 1:
        imgs = F.avg_pool2d(imgs, int_factor)
    imgs = F.interpolate(imgs, [image_size, image_size], mode="bilinear")

    if len(ss) == 3:
        imgs = imgs[0]
    return imgs


def to_2tuple(x):
    return (x, x) if not isinstance(x, tuple) else x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim,
        init_values=1e-5,
        inplace=False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        ffn_bias=True,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=Attention,
        ffn_layer=Mlp,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        init_values=None,
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer=Mlp,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        blocks_list = [
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                ffn_bias=ffn_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0, h0 = w // self.patch_size, h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx, sy = (
                float(w0 + self.interpolate_offset) / M,
                float(h0 + self.interpolate_offset) / M,
            )
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = (
            nn.functional.interpolate(
                patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                mode="bicubic",
                antialias=self.interpolate_antialias,
                # align_corners=False,
                **kwargs,
            )
            .permute(0, 2, 3, 1)
            .view(1, -1, dim)
        )
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(x.dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.size(0), -1, -1), x[:, 1:]),
                dim=1,
            )
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, is_training=False, masks=None):
        # x = x.movedim(-1, 1)  # Move channel dimension to the front
        ret = self.forward_features(x, masks)
        return ret if is_training else self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(patch_size=14, num_register_tokens=0, **kwargs):
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=8 / 3,
        num_register_tokens=num_register_tokens,
        ffn_layer=SwiGLUFFN,
        **kwargs,
    )


# Variants with registers based on `https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py`.
# Note the interpolate args for positional encoding are different to variants without registers.

vit_small_reg = partial(
    vit_small,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)

vit_base_reg = partial(
    vit_base,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)

vit_large_reg = partial(
    vit_large,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)

vit_giant2_reg = partial(
    vit_giant2,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
)


model_list = {
    "dinov2_vits14": vit_small,
    "dinov2_vitb14": vit_base,
    "dinov2_vitl14": vit_large,
    "dinov2_vitg14": vit_giant2,
    "dinov2_vits14_reg": vit_small_reg,
    "dinov2_vitb14_reg": vit_base_reg,
    "dinov2_vitl14_reg": vit_large_reg,
    "dinov2_vitg14_reg": vit_giant2_reg,
}


class MinDino(nn.Module):
    def __init__(
        self,
        model_version: str | None,
        model_size: int = 224,
        out: Literal["features", "class", "both", "regs", "dict"] = "both",
        requires_grad: bool = False,
        reshape: bool = True,
        custom_model: dict | None = None,
    ):
        super().__init__()
        self.requires_grad = requires_grad
        self.out = out
        self.model_size = model_size
        self.model_version = model_version
        self.reshape = reshape

        if model_version is not None:
            dino: nn.Module = torch.hub.load("facebookresearch/dinov2", model_version)
            self.model: nn.Module = model_list[model_version](img_size=model_size)
            self.model.load_state_dict(dino.state_dict())
        else:
            assert custom_model is not None, "Either model_version or custom_model must be provided"
            self.model: nn.Module = DinoVisionTransformer(**custom_model)

        if requires_grad:
            self.model.requires_grad_(True)
            self.model.mask_token.requires_grad_(False)
            self.model.train()
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(
        self, imgs: Float[torch.Tensor, "B H W C"]
    ) -> tuple[Float[torch.Tensor, "B h' w' D"], Float[torch.Tensor, "B D"]]:
        assert imgs.min() >= -1.01
        assert imgs.max() <= 1.01
        assert len(imgs.shape) == 4
        assert imgs.size(-1) == 3, "Input images must be channel-last (B H W C) with 3 channels"

        imgs = imgs.movedim(-1, 1)  # Move channel dimension to the front

        imgs = better_resize(imgs, self.model_size)

        imgs = (imgs + 1.0) / 2.0
        # copied from transformers preprocessor
        imgs = imgs - torch.tensor([0.485, 0.456, 0.406], device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
        imgs = imgs / torch.tensor([0.229, 0.224, 0.225], device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)

        # is_training only results in getting the complete output (cls, features, regs)
        with torch.no_grad() if not self.requires_grad else contextlib.nullcontext():
            out_dict = self.model(imgs, is_training=True)

        features = out_dict["x_norm_patchtokens"]

        if self.reshape:
            features = einops.rearrange(
                features,
                "b (h w) c -> b h w c",
                h=self.model_size // self.model.patch_size,
                w=self.model_size // self.model.patch_size,
            )

        if self.out == "features":
            return features  # [B H W D]
        elif self.out == "class":
            return out_dict["x_norm_clstoken"].unsqueeze(1)  # [B 1 D]
        elif self.out == "both":
            # mostly for backward compatibility
            return features, out_dict["x_norm_clstoken"].unsqueeze(1)  # [B H W D], [B 1 D]
        elif self.out == "regs":
            return out_dict["x_norm_regtokens"]  # [B 4 D]
        elif self.out == "dict":
            return out_dict
        else:
            raise ValueError(f"Unknown output type: {self.out}")


if __name__ == "__main__":
    # this will test the similarity and l1 loss between the output of dino and our reimplementation

    for model_name, model_fn in model_list[3:]:
        dino = torch.hub.load("facebookresearch/dinov2", model_name).cuda()
        model = model_fn().cuda()
        model.load_state_dict(dino.state_dict())

        for h, w in [(224, 224), (140, 140), (448, 448)]:
            image = torch.randn(1, 3, h, w).cuda()
            output_dino = dino(image)
            out_ours = model(image)

            cos_sim = F.cosine_similarity(output_dino, out_ours).item()
            l1_loss = F.l1_loss(output_dino, out_ours).item()
            print(f"Similarity between output_dino and out_ours: {cos_sim}")
            print(f"L1 distance between output_dino and out_ours: {l1_loss}")

            assert cos_sim > 0.99
            assert l1_loss < 0.01
