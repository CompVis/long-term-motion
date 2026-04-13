import math
from functools import partial

import einops
import torch
from jaxtyping import Float
from torch import nn
from tqdm import trange

from zipmo.blocks import FeedForwardBlock, InputMLP, Level, RMSNorm, TransformerLayer
from zipmo.dino import MinDino
from zipmo.rope import centers, make_axial_pos_2d
from zipmo.vae import ZipMoVAE


class CondTokenConcatMerge(nn.Module):
    def __init__(self, in_features: int, out_features: int, cond_features: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features, bias=False)
        self.cond_proj = nn.Linear(cond_features, out_features, bias=False)

    def forward(self, x, pos, extra_tokens, extra_pos, **kwargs):
        return torch.cat((self.proj(x), self.cond_proj(extra_tokens)), dim=1), torch.cat((pos, extra_pos), dim=1)


class CondTokenConcatSplit(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_extra_tokens: int):
        super().__init__()
        self.num_extra_tokens = num_extra_tokens
        self.proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, **kwargs):
        return self.proj(x[:, : -self.num_extra_tokens])


class MappingNetwork(nn.Module):
    def __init__(self, depth, width, d_ff):
        super().__init__()
        self.in_norm = RMSNorm(width)
        self.blocks = nn.ModuleList([FeedForwardBlock(width, d_ff) for _ in range(depth)])
        self.out_norm = RMSNorm(width)

    def forward(self, x):
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x):
        features = 2 * math.pi * x @ self.weight.T
        return torch.cat((features.cos(), features.sin()), dim=-1)


class ZipMoPlanner(nn.Module):
    """
    Clean generator implementation for the released `traj_gen_v2` model.

    This module intentionally supports only the configuration used in
    `configs/experiment/traj_gen_v2.yaml`:
    - 1 x 16 x 16 latent grid with 16 latent channels
    - 24-layer 1024-dim transformer backbone
    - DINOv2-B/14 start-frame tokens
    - 8 endpoint conditions with Poisson condition dropping during training
    """

    def __init__(
        self,
        n_cond: int,
        poisson_rate: float | None,
        vae: ZipMoVAE,
        depth: int = 24,
        width: int = 1024,
        d_cross: int = 768,
        d_cond_norm: int = 256,
        vae_shift: float = -0.175,
        vae_scale: float = 11.6,
    ):
        super().__init__()

        self.vae = vae

        self.depth = depth
        self.width = width
        self.d_cross = d_cross
        self.d_cond_norm = d_cond_norm

        self.grid_size = (1, 16, 16)
        self.latent_dim = 16
        self.n_cond = n_cond
        self.poisson_rate = float(poisson_rate) if poisson_rate is not None else None
        self.cfg_scale = float(1)

        self.img_embedder = MinDino("dinov2_vitb14_reg", out="features", requires_grad=False)
        self.track_cond_emb = InputMLP(in_features=3, dim=d_cross, random_fourier=True, theta=10.0 * 3.14)
        self.time_emb = FourierFeatures(1, 256)
        self.time_in_proj = nn.Linear(256, 256, bias=False)
        self.mapping = MappingNetwork(depth=2, width=256, d_ff=768)
        self.in_proj = CondTokenConcatMerge(in_features=self.latent_dim, out_features=1024, cond_features=d_cross)

        self.backbone = Level(
            [
                TransformerLayer(
                    d_model=width,
                    d_cross=d_cross,
                    d_cond_norm=d_cond_norm,
                    self_rope_mode="3d_10",
                    cross_rope_mode="3d_10",
                )
                for _ in range(depth)
            ]
        )

        self.out_proj = CondTokenConcatSplit(
            in_features=1024,
            out_features=self.latent_dim,
            num_extra_tokens=self.grid_size[1] * self.grid_size[2],
        )

        self.learnable_emb_dino = nn.Parameter(torch.randn(d_cross) * 0.02)

        self.vae_shift = torch.tensor(vae_shift, dtype=torch.float64)
        self.vae_scale = torch.tensor(vae_scale, dtype=torch.float64)

        print(f"{self.vae_scale=}", flush=True)
        print(f"{self.vae_shift=}", flush=True)

    def get_pos(self, x: Float[torch.Tensor, "b l c"]) -> Float[torch.Tensor, "b l 3"]:
        batch_size = x.shape[0]
        num_latent_tokens = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        assert x.shape[1] == num_latent_tokens, f"Expected {num_latent_tokens} latent tokens, got {x.shape[1]}"

        time_pos = einops.repeat(
            centers(-1, 1, self.grid_size[0], device=x.device),
            "t -> (t h w) 1",
            h=self.grid_size[1],
            w=self.grid_size[2],
        )
        spatial_pos = einops.repeat(
            make_axial_pos_2d(self.grid_size[1], self.grid_size[2], device=x.device),
            "(h w) c -> (t h w) c",
            t=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
        )
        pos = torch.cat((time_pos, spatial_pos), dim=-1)
        return pos.unsqueeze(0).expand(batch_size, -1, -1)

    def _time_condition(self, t: Float[torch.Tensor, "b"]) -> Float[torch.Tensor, "b c"]:
        time_features = self.time_emb(t[:, None])
        time_features = self.time_in_proj(time_features)
        return self.mapping(time_features)

    def get_frame_embed(self, start_frame, time_idx: int):
        frame_embed = self.img_embedder(start_frame)
        batch_size, height, width, channels = frame_embed.shape
        frame_embed = einops.rearrange(frame_embed, "b h w c -> b (h w) c")
        time_pos = centers(-1, 1, self.grid_size[0], device=frame_embed.device)[time_idx].expand(height * width, 1)
        pos = torch.cat(
            (time_pos, make_axial_pos_2d(height, width, device=frame_embed.device)),
            dim=-1,
        )
        return frame_embed, pos.unsqueeze(0).expand(batch_size, -1, -1)

    def get_start_frame_embed(self, start_frame):
        frame_embed, frame_pos = self.get_frame_embed(start_frame, time_idx=0)
        frame_embed = frame_embed + einops.rearrange(self.learnable_emb_dino, "c -> 1 1 c")
        return frame_embed, frame_pos

    def get_static_conditioning(self, start_frame, track_conds):
        batch_size = start_frame.shape[0]
        start_embed, start_pos = self.get_start_frame_embed(start_frame)

        start_cond = track_conds[..., :2]
        end_cond = track_conds[..., 2:]
        track_cond_emb = self.track_cond_emb(end_cond)
        num_cond = track_cond_emb.shape[1]

        pos_cross = torch.cat(
            (
                centers(-1, 1, self.grid_size[0], device=start_cond.device)[0].expand(batch_size, num_cond, 1),
                start_cond,
            ),
            dim=-1,
        )

        if self.training and self.poisson_rate is not None:
            # drop out some tracks
            n_drop = torch.poisson(torch.full((batch_size,), self.poisson_rate, device=start_cond.device))
            n_drop = n_drop.clamp(0, num_cond).long()
            keep_mask = torch.arange(num_cond, device=start_cond.device)[None] < (num_cond - n_drop[:, None])
            track_cond_emb = track_cond_emb * keep_mask[..., None]
            pos_cross = pos_cross * keep_mask[..., None]

        if not self.training and self.poisson_rate is not None:
            # support padding if the model was trained on dropped pokes
            n_pad = self.n_cond - num_cond
            track_cond_emb = torch.cat(
                [
                    track_cond_emb,
                    torch.zeros(
                        batch_size,
                        n_pad,
                        track_cond_emb.shape[-1],
                        device=track_cond_emb.device,
                        dtype=track_cond_emb.dtype,
                    ),
                ],
                dim=1,
            )
            pos_cross = torch.cat(
                [
                    pos_cross,
                    torch.zeros(batch_size, n_pad, 3, device=pos_cross.device, dtype=pos_cross.dtype),
                ],
                dim=1,
            )

        assert (
            track_cond_emb.shape[1] == self.n_cond
        ), f"Expected track_cond_emb with {self.n_cond} tokens, got {track_cond_emb.shape[1]}"

        return {
            "extra_tokens": start_embed,
            "extra_pos": start_pos,
            "x_cross": track_cond_emb,
            "pos_cross": pos_cross,
        }

    def get_static_unconditioning(self, start_frame):
        batch_size = start_frame.shape[0]
        start_embed, start_pos = self.get_start_frame_embed(start_frame)
        return {
            "extra_tokens": start_embed,
            "extra_pos": start_pos,
            "x_cross": torch.zeros(
                batch_size,
                self.n_cond,
                self.d_cross,
                dtype=start_frame.dtype,
                device=start_frame.device,
            ),
            "pos_cross": torch.zeros(
                (batch_size, self.n_cond, 3),
                dtype=start_frame.dtype,
                device=start_frame.device,
            ),
        }

    def get_track_cond(self, tracks: Float[torch.Tensor, "b n t 2"]):
        batch_size, num_tracks, num_points, _ = tracks.shape
        assert self.n_cond <= num_tracks, "traj_gen_v2 samples endpoint conditions without replacement"

        traj_idx = torch.stack(
            [torch.randperm(num_tracks, device=tracks.device)[: self.n_cond] for _ in range(batch_size)]
        )
        batch_idx = einops.repeat(
            torch.arange(batch_size, device=tracks.device),
            "b -> b n_cond",
            n_cond=self.n_cond,
        )
        end_t = torch.full((batch_size, self.n_cond), num_points - 1, device=tracks.device)

        start_points = tracks[batch_idx, traj_idx, 0]
        end_points = tracks[batch_idx, traj_idx, end_t]
        end_time = end_t.unsqueeze(-1).float() / (num_points - 1)
        return torch.cat((start_points, end_points, end_time), dim=-1), traj_idx

    def normalize_latents(self, latents: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        """Normalize latents to zero mean and unit variance per channel using precomputed dataset statistics."""
        dtype = latents.dtype
        latents.to(torch.float64)
        if self.vae_shift is not None:
            latents = latents - self.vae_shift.to(latents.device).view(1, 1, -1)

        if self.vae_scale is not None:  # scale variance to 0.5
            latents = latents * (math.sqrt(0.5) / self.vae_scale.sqrt().to(latents.device))

        return latents.to(dtype)

    def denormalize_latents(self, latents: Float[torch.Tensor, "B N C"]) -> Float[torch.Tensor, "B N C"]:
        """Un-normalize latents to original distribution using precomputed dataset statistics."""
        dtype = latents.dtype
        latents.to(torch.float64)

        if self.vae_scale is not None:
            latents = latents * (self.vae_scale.sqrt().to(latents.device) / math.sqrt(0.5))

        if self.vae_shift is not None:
            latents = latents + self.vae_shift.to(latents.device).view(1, 1, -1)

        return latents.to(dtype)

    def _predict_velocity(self, x_t, t, static_cond):
        cond_norm = self._time_condition(t.to(x_t.dtype))
        pos = self.get_pos(x_t)
        x_t, pos = self.in_proj(x_t, pos, **static_cond)
        x_t = self.backbone(x_t, pos=pos, cond_norm=cond_norm, **static_cond)
        return self.out_proj(x_t)

    def _rf_loss(self, latents, start_frame, track_conds):
        batch_size = latents.shape[0]
        t = torch.sigmoid(torch.randn((batch_size,), device=latents.device, dtype=latents.dtype))
        t_expanded = t.view(batch_size, 1, 1)
        noise = torch.randn_like(latents)
        x_t = (1 - t_expanded) * latents + t_expanded * noise

        static_cond = self.get_static_conditioning(start_frame=start_frame, track_conds=track_conds)
        velocity = self._predict_velocity(x_t, t, static_cond)
        return ((noise - latents - velocity) ** 2).mean(dim=(1, 2))

    def forward(
        self,
        tracks_enc_yx: Float[torch.Tensor, "b n t 2"],
        start_frame: Float[torch.Tensor, "b h w c"],
        **kwargs,
    ):
        with torch.no_grad():
            encoded = self.vae.encode(tracks_enc_yx, start_frame=start_frame)
            latents = self.normalize_latents(encoded.mean)
            track_conds, _ = self.get_track_cond(tracks_enc_yx)

        rf_loss = self._rf_loss(latents, start_frame=start_frame, track_conds=track_conds)
        return dict(rf_loss=rf_loss), dict()

    @torch.no_grad()
    def sample(
        self,
        z: Float[torch.Tensor, "b l c"],
        points_per_traj: int,
        query_pos: Float[torch.Tensor, "b n 2"],
        track_conds: Float[torch.Tensor, "b n_cond 5"],
        start_frame: Float[torch.Tensor, "b h w c"],
        sample_steps: int = 50,
        decode_latent: bool = True,
        decode_dense: bool = False,
    ):
        batch_size = z.shape[0]
        dt = torch.full((batch_size, 1, 1), 1.0 / sample_steps, device=z.device, dtype=z.dtype)

        static_cond = self.get_static_conditioning(start_frame=start_frame, track_conds=track_conds)
        static_uncond = self.get_static_unconditioning(start_frame=start_frame) if self.cfg_scale > 1.0 else None

        latents = z
        for step in trange(sample_steps, 0, -1, desc="Sampling"):
            t = torch.full((batch_size,), step / sample_steps, device=z.device, dtype=z.dtype)
            cond_velocity = self._predict_velocity(latents, t, static_cond)

            if static_uncond is None:
                latents = latents - dt * cond_velocity
                continue

            uncond_velocity = self._predict_velocity(latents, t, static_uncond)
            guided_velocity = uncond_velocity + self.cfg_scale * (cond_velocity - uncond_velocity)
            latents = latents - dt * guided_velocity

        if not decode_latent:
            return latents

        latents = self.denormalize_latents(latents)

        if decode_dense:
            return self.vae.decode_dense(
                latents=latents,
                points_per_track=points_per_traj,
                start_frame=start_frame,
            )

        return self.vae.decode(
            latents=latents,
            query_pos=query_pos,
            points_per_track=points_per_traj,
            start_frame=start_frame,
        )


ZipMoPlanner_Sparse: ZipMoPlanner = partial(ZipMoPlanner, n_cond=16, poisson_rate=2.5)  # type: ignore
ZipMoPlanner_Dense: ZipMoPlanner = partial(ZipMoPlanner, n_cond=40, poisson_rate=None)  # type: ignore


class ZipMoPlanner_Libero(ZipMoPlanner):
    def __init__(
        self,
        vae: ZipMoVAE,
        n_cond: int = 1,
        poisson_rate: float | None = None,
        num_views: int = 2,
        text_enc_dim: int = 768,
        use_t_input: bool = False,
        c_dropout: float = 0.1,
        c_dropout_start_frame: float = 0.0,
        text_enc_depth: int = 6,
        **kwargs,
    ):
        super().__init__(n_cond=n_cond, poisson_rate=poisson_rate, vae=vae, **kwargs)

        self.num_views = num_views
        self.use_t_input = use_t_input
        self.c_dropout = float(c_dropout)
        self.c_dropout_start_frame = float(c_dropout_start_frame)
        self.text_encoder = Level(
            [
                TransformerLayer(
                    d_model=text_enc_dim,
                    use_ca=False,
                    self_rope_mode="1d",
                )
                for _ in range(text_enc_depth)
            ]
        )

        self.backbone = Level(
            [
                TransformerLayer(
                    d_model=self.width,
                    d_cross=self.d_cross,
                    d_cond_norm=self.d_cond_norm,
                    cross_rope_mode="none",
                )
                for _ in range(self.depth)
            ]
        )

        self.img_emb_dim = self.d_cross
        self.learnable_emb_dino = nn.Parameter(torch.randn(num_views, self.img_emb_dim) * 0.02, requires_grad=True)
        self.learnable_emb_grid_tokens = nn.Parameter(torch.randn(num_views, self.width) * 0.02, requires_grad=True)
        self.ca_mask_token = nn.Parameter(torch.randn(self.d_cross) * 0.02, requires_grad=True)

        if use_t_input:
            self.start_t_mapping = InputMLP(
                in_features=1,
                dim=self.d_cross,
                random_fourier=True,
                theta=10.0 * math.pi,
            )

        # In the Libero multi-view setup we append DINO tokens from all views.
        # `out_proj` must therefore strip `num_views * (nh * nw)` extra tokens.
        self.out_proj = CondTokenConcatSplit(
            in_features=self.width,
            out_features=self.latent_dim,
            num_extra_tokens=self.num_views * self.grid_size[1] * self.grid_size[2],
        )

        n_latent_tokens = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.val_shape = (self.num_views * n_latent_tokens, self.latent_dim)

    def get_pos(self, x: Float[torch.Tensor, "b l c"]) -> Float[torch.Tensor, "b l 3"]:
        tokens_per_view = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        expected = self.num_views * tokens_per_view
        if x.shape[1] != expected:
            raise ValueError(f"Expected {expected} latent tokens, got {x.shape[1]}")
        pos_single = super().get_pos(x[:, :tokens_per_view])
        return einops.repeat(pos_single, "b l c -> b (v l) c", v=self.num_views)

    def get_start_frame_embed(self, start_frame):
        if start_frame.ndim == 4:
            return super().get_start_frame_embed(start_frame)
        if start_frame.ndim != 5:
            raise ValueError(f"Expected start_frame with shape [B, V, H, W, C], got {tuple(start_frame.shape)}")

        batch_size, num_views, _, _, _ = start_frame.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {num_views}")

        start_frame_flat = einops.rearrange(start_frame, "b v h w c -> (b v) h w c")
        start_frame_embed = self.img_embedder(start_frame_flat.clip(-1, 1))
        _, nh, nw, _ = start_frame_embed.shape

        start_frame_embed = einops.rearrange(
            start_frame_embed, "(b v) nh nw c -> b v (nh nw) c", b=batch_size, v=num_views
        )
        start_frame_embed = start_frame_embed + einops.rearrange(self.learnable_emb_dino, "v c -> 1 v 1 c")
        start_frame_embed = einops.rearrange(start_frame_embed, "b v l c -> b (v l) c")

        start_frame_pos = torch.cat(
            (
                torch.full((nh * nw, 1), -1.0, dtype=start_frame.dtype, device=start_frame.device),
                make_axial_pos_2d(nh, nw, device=start_frame.device),
            ),
            dim=-1,
        )
        start_frame_pos = einops.repeat(start_frame_pos, "l c -> b (v l) c", b=batch_size, v=num_views)
        return start_frame_embed, start_frame_pos

    def _append_start_t(self, cond: dict[str, torch.Tensor], start_t: Float[torch.Tensor, "b"] | None):
        if not self.use_t_input:
            return cond
        if start_t is None:
            raise ValueError("start_t must be provided when use_t_input=True.")

        start_t_emb = self.start_t_mapping(einops.rearrange(start_t, "b -> b 1 1"))
        cond["x_cross"] = torch.cat((cond["x_cross"], start_t_emb), dim=1)

        return cond

    def get_static_conditioning(
        self,
        start_frame: Float[torch.Tensor, "b v h w c"],
        track_conds: Float[torch.Tensor, "b n_cond 5"],
        txt_emb: Float[torch.Tensor, "b t c"],
        start_t: Float[torch.Tensor, "b"] | None = None,
    ):
        cond = super().get_static_conditioning(start_frame=start_frame, track_conds=track_conds)
        txt_emb = txt_emb.to(device=cond["x_cross"].device, dtype=cond["x_cross"].dtype)
        batch_size, num_txt_tokens, _ = txt_emb.shape
        txt_pos = torch.arange(num_txt_tokens, device=txt_emb.device)[None, :, None].expand(batch_size, -1, 1)
        cond["x_cross"] = self.text_encoder(x=txt_emb, pos=txt_pos)

        _, token_count, _ = cond["extra_tokens"].shape
        tokens_per_view = token_count // self.num_views
        cond["learned_pos_emb"] = einops.repeat(
            self.learnable_emb_grid_tokens,
            "v c -> b (v l) c",
            b=batch_size,
            l=tokens_per_view,
        )

        if self.training and self.c_dropout > 0.0:
            keep_mask = (torch.rand(batch_size, device=cond["x_cross"].device) >= self.c_dropout)[:, None, None]
            mask_token = einops.rearrange(
                self.ca_mask_token.to(dtype=cond["x_cross"].dtype, device=cond["x_cross"].device),
                "c -> 1 1 c",
            ).expand(batch_size, 1, -1)
            cond["x_cross"] = torch.where(keep_mask, cond["x_cross"], mask_token)

        cond = self._append_start_t(cond, start_t=start_t)

        if self.training and self.c_dropout_start_frame > 0.0:
            keep_mask = (torch.rand(batch_size, device=cond["x_cross"].device) >= self.c_dropout_start_frame)[
                :, None, None
            ]
            mask_token = einops.rearrange(
                self.ca_mask_token.to(dtype=cond["extra_tokens"].dtype, device=cond["extra_tokens"].device),
                "c -> 1 1 c",
            ).expand(batch_size, 1, -1)
            cond["extra_tokens"] = torch.where(keep_mask, cond["extra_tokens"], mask_token)

        return cond

    def get_static_unconditioning(
        self,
        start_frame: Float[torch.Tensor, "b v h w c"],
        start_t: Float[torch.Tensor, "b"] | None = None,
    ):
        cond = super().get_static_unconditioning(start_frame=start_frame)
        batch_size = start_frame.shape[0]

        _, token_count, _ = cond["extra_tokens"].shape
        if token_count % self.num_views != 0:
            raise ValueError(f"Token count ({token_count}) must be divisible by num_views ({self.num_views}).")
        tokens_per_view = token_count // self.num_views
        cond["learned_pos_emb"] = einops.repeat(
            self.learnable_emb_grid_tokens,
            "v c -> b (v l) c",
            b=batch_size,
            l=tokens_per_view,
        )

        cond["x_cross"] = einops.rearrange(
            self.ca_mask_token.to(dtype=cond["extra_tokens"].dtype, device=cond["extra_tokens"].device),
            "c -> 1 1 c",
        ).expand(batch_size, 1, -1)
        cond["pos_cross"] = torch.zeros(
            (batch_size, 1, 3),
            dtype=cond["x_cross"].dtype,
            device=cond["x_cross"].device,
        )

        cond = self._append_start_t(cond, start_t=start_t)
        return cond

    def _predict_velocity(self, x_t, t, static_cond):
        cond_norm = self._time_condition(t.to(x_t.dtype))
        pos = self.get_pos(x_t)
        x_t, pos = self.in_proj(x_t, pos, **static_cond)

        learned_pos_emb = static_cond.get("learned_pos_emb")
        if learned_pos_emb is not None:
            learned_pos_emb = learned_pos_emb.to(device=x_t.device, dtype=x_t.dtype)
            n_grid = learned_pos_emb.shape[1]
            x_t = torch.cat((x_t[:, :n_grid] + learned_pos_emb, x_t[:, n_grid:]), dim=1)

        x_t = self.backbone(x_t, pos=pos, cond_norm=cond_norm, **static_cond)
        return self.out_proj(x_t)

    def _rf_loss(
        self,
        latents: Float[torch.Tensor, "b l c"],
        start_frame: Float[torch.Tensor, "b v h w c"],
        track_conds: Float[torch.Tensor, "b n_cond 5"],
        txt_emb: Float[torch.Tensor, "b t c"],
        start_t: Float[torch.Tensor, "b"] | None = None,
    ):
        batch_size = latents.shape[0]
        t = torch.sigmoid(torch.randn((batch_size,), device=latents.device, dtype=latents.dtype))
        t_expanded = t.view(batch_size, 1, 1)
        noise = torch.randn_like(latents)
        x_t = (1 - t_expanded) * latents + t_expanded * noise

        static_cond = self.get_static_conditioning(
            start_frame=start_frame,
            track_conds=track_conds,
            txt_emb=txt_emb,
            start_t=start_t,
        )
        velocity = self._predict_velocity(x_t, t, static_cond)
        return ((noise - latents - velocity) ** 2).mean(dim=(1, 2))

    def forward(
        self,
        tracks_enc_yx: Float[torch.Tensor, "b v n t 2"],
        start_frame: Float[torch.Tensor, "b v h w c"],
        task_emb_bert: Float[torch.Tensor, "b c"],
        start_t: Float[torch.Tensor, "b"] | None = None,
        **kwargs,
    ):
        del kwargs
        txt_emb = einops.rearrange(task_emb_bert, "b c -> b 1 c").to(device=start_frame.device)

        batch_size, num_views, _, _, _ = tracks_enc_yx.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {num_views}")

        with torch.no_grad():
            track_conds, _ = self.get_track_cond(tracks_enc_yx[:, 0, ...])

            tracks_enc_yx_flat = einops.rearrange(tracks_enc_yx, "b v n t c -> (b v) n t c")
            start_frame_flat = einops.rearrange(start_frame, "b v h w c -> (b v) h w c")
            encoded = self.vae.encode(tracks_enc_yx_flat, start_frame=start_frame_flat)
            latents = self.normalize_latents(encoded.mean)
            latents = einops.rearrange(latents, "(b v) l c -> b (v l) c", b=batch_size, v=num_views)

        rf_loss = self._rf_loss(
            latents=latents,
            start_frame=start_frame,
            track_conds=track_conds,
            txt_emb=txt_emb,
            start_t=start_t,
        )
        return dict(rf_loss=rf_loss), dict()

    @torch.no_grad()
    def sample(
        self,
        z: Float[torch.Tensor, "b l c"],
        track_conds: Float[torch.Tensor, "b n_cond 5"],
        txt_emb: Float[torch.Tensor, "b t c"],
        points_per_traj: int | None = None,
        query_pos: Float[torch.Tensor, "b v n 2"] | None = None,
        start_frame: Float[torch.Tensor, "b v h w c"] | None = None,
        sample_steps: int = 50,
        return_list: bool = False,
        decode_latent: bool = True,
        start_t: Float[torch.Tensor, "b"] | None = None,
        **kwargs,
    ):
        del kwargs

        if start_frame is None:
            raise ValueError("start_frame must be provided.")

        batch_size = z.shape[0]
        txt_emb = txt_emb.to(device=z.device)

        dt = torch.full((batch_size, 1, 1), 1.0 / sample_steps, device=z.device, dtype=z.dtype)

        static_cond = self.get_static_conditioning(
            start_frame=start_frame,
            track_conds=track_conds,
            txt_emb=txt_emb,
            start_t=start_t,
        )
        static_uncond = (
            self.get_static_unconditioning(start_frame=start_frame, start_t=start_t) if self.cfg_scale > 1.0 else None
        )

        latents = z
        trajectory = [latents] if return_list else None
        for step in range(sample_steps, 0, -1):
            t = torch.full((batch_size,), step / sample_steps, device=z.device, dtype=z.dtype)
            cond_velocity = self._predict_velocity(latents, t, static_cond)

            if static_uncond is None:
                latents = latents - dt * cond_velocity
            else:
                uncond_velocity = self._predict_velocity(latents, t, static_uncond)
                guided_velocity = uncond_velocity + self.cfg_scale * (cond_velocity - uncond_velocity)
                latents = latents - dt * guided_velocity

            if return_list:
                trajectory.append(latents)

        if return_list:
            return trajectory

        if not decode_latent:
            return latents

        latents = self.denormalize_latents(latents)
        if query_pos is None:
            raise ValueError("query_pos must be provided for decoding.")
        if points_per_traj is None:
            raise ValueError("points_per_traj must be provided for decoding.")

        latents = einops.rearrange(latents, "b (v l) c -> (b v) l c", v=self.num_views)
        start_frame = einops.rearrange(start_frame, "b v h w c -> (b v) h w c")
        query_pos = einops.rearrange(query_pos, "b v n c -> (b v) n c")

        tracks = self.vae.decode(
            latents=latents,
            query_pos=query_pos,
            points_per_track=points_per_traj,
            start_frame=start_frame,
        )
        return einops.rearrange(tracks, "(b v) n t c -> b v n t c", v=self.num_views)


ZipMoPlanner_Libero_ATM: ZipMoPlanner_Libero = partial(ZipMoPlanner_Libero, use_t_input=False)  # type: ignore
ZipMoPlanner_Libero_TraMoE: ZipMoPlanner_Libero = partial(ZipMoPlanner_Libero, use_t_input=True)  # type: ignore
