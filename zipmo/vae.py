from collections import defaultdict
from dataclasses import dataclass

import einops
import torch
import torch.distributed as dist
import wandb
from jaxtyping import Float
from torch import nn
from tqdm import tqdm

from zipmo.blocks import InputMLP, Level, OutputMLP, TransformerLayer
from zipmo.dino import MinDino
from zipmo.rope import centers, make_axial_pos_2d


def chunk_grid_strided(grid, chunk_size=8):
    """
    Chunk a (B, H, W, 2) or (H, W, 2) grid into strided batches.
    Each chunk takes every chunk_size-th point starting from a different offset.

    This maximizes coverage: each chunk spans nearly the full [-1, 1] range
    by sampling every chunk_size-th point across the entire grid.

    For example with chunk_size=8 and 64x64 grid:
    - Chunk 0: indices [0, 8, 16, 24, 32, 40, 48, 56] in both x and y
    - Chunk 1: indices [1, 9, 17, 25, 33, 41, 49, 57] in both x and y
    - ... and so on

    Args:
        grid: torch.Tensor of shape (B, H, W, 2) or (H, W, 2) containing (x, y) coordinates
        chunk_size: stride size (default: 8)

    Returns:
        torch.Tensor of shape (B, num_chunks, chunk_size, chunk_size, 2) if input has batch dim
        or (num_chunks, chunk_size, chunk_size, 2) if input has no batch dim
        where num_chunks = chunk_size * chunk_size
    """
    has_batch = grid.ndim == 4

    if not has_batch:
        grid = grid.unsqueeze(0)  # Add batch dim temporarily

    B, H, W, C = grid.shape

    assert H % chunk_size == 0 and W % chunk_size == 0, (
        f"Grid dimensions ({H}, {W}) must be divisible by chunk_size ({chunk_size})"
    )

    chunks_per_dim = H // chunk_size

    # First rearrange to interleave offsets with strided points
    # The pattern is: [0, 8, 16, ...], [1, 9, 17, ...], etc.
    # So we want: points at positions (offset + k*chunk_size) for each offset
    # This means: [offset0: 0,8,16,...], [offset1: 1,9,17,...], etc.
    chunks = einops.rearrange(
        grid,
        "b (points_h offset_h) (points_w offset_w) c -> b (offset_h offset_w) points_h points_w c",
        offset_h=chunks_per_dim,
        offset_w=chunks_per_dim,
        points_h=chunk_size,
        points_w=chunk_size,
    )

    if not has_batch:
        chunks = chunks.squeeze(0)  # Remove batch dim if input didn't have it

    return chunks


def assemble_chunks_strided(chunks, grid_size: int):
    """
    Assemble strided chunks back into the original full grid.

    Args:
        chunks: torch.Tensor of shape (B, num_chunks, chunk_size, chunk_size, 2)
                or (num_chunks, chunk_size, chunk_size, 2)
        chunk_size: stride size used in chunking (default: 8)

    Returns:
        torch.Tensor of shape (B, grid_height, grid_width, 2) if input has batch dim
        or (grid_height, grid_width, 2) if input has no batch dim
    """
    has_batch = chunks.ndim == 5

    if not has_batch:
        chunks = chunks.unsqueeze(0)  # Add batch dim temporarily

    B, num_chunks, chunk_size, _, C = chunks.shape

    chunks_per_dim = grid_size // chunk_size

    assert num_chunks == chunks_per_dim * chunks_per_dim, (
        f"Number of chunks ({num_chunks}) should equal chunks_per_dim^2 ({chunks_per_dim}^2)"
    )

    # Reverse the rearrange operation
    grid = einops.rearrange(
        chunks,
        "b (offset_h offset_w) points_h points_w c -> b (points_h offset_h) (points_w offset_w) c",
        offset_h=chunks_per_dim,
        offset_w=chunks_per_dim,
    )

    if not has_batch:
        grid = grid.squeeze(0)  # Remove batch dim if input didn't have it

    return grid


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE sampling.
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Calculate KL divergence between learned distribution and standard normal.
    """
    return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)


@dataclass(frozen=True)
class EncoderOutput:
    mean: Float[torch.Tensor, "B n_tokens C"]
    logvar: Float[torch.Tensor, "B n_tokens C"]
    latent_pos: Float[torch.Tensor, "B n_tokens 3"]


class TrajRegressorDecoderMAE(nn.Module):
    # annotate the buffers
    pos_cross: Float[torch.Tensor, "1 n_patches 3"]

    def __init__(
        self,
        d_model: int = 768,
        d_cross: int = 768,
        depth: int = 12,
        latent_dim: int = 16,
        img_feat_size=(16, 16),
        criterion: nn.Module = nn.L1Loss(),
    ):
        super().__init__()

        self.backbone = Level([TransformerLayer(d_model=d_model, d_cross=d_cross) for _ in range(depth)])
        self.trajs_out = OutputMLP(d_model, 2)
        self.decode_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.latents_in = nn.Linear(latent_dim, d_model)
        self.criterion = criterion

        H, W = img_feat_size
        time_cross = -1.0
        pos_cross = torch.cat([torch.full((H * W, 1), time_cross), make_axial_pos_2d(H, W)], dim=-1)
        self.register_buffer("pos_cross", pos_cross)  # [1, n_tokens, 3]

    def forward(
        self,
        latents: Float[torch.Tensor, "B n_tokens C"],
        tracks_yx: Float[torch.Tensor, "B N T 2"],
        latent_pos: Float[torch.Tensor, "B n_tokens 3"],
        start_emb: Float[torch.Tensor, "B (H W) C"],
        **data_kwargs,
    ) -> Float[torch.Tensor, "B"]:
        B, num_trajs, points_per_traj, C = tracks_yx.shape
        start_pos = tracks_yx[:, :, 0, :]  # (B, N, 2)
        decoded_tracks = self.decode_tracks(
            latents=latents,
            points_per_traj=points_per_traj,
            num_trajs=num_trajs,
            query_pos=start_pos,
            latent_pos=latent_pos,
            start_emb=start_emb,
        )

        return self.criterion(decoded_tracks, tracks_yx)

    def get_decode_pos(
        self,
        latents,
        points_per_traj: int,
        num_trajs: int,
        query_pos: Float[torch.Tensor, "B N 2"],
    ) -> Float[torch.Tensor, "B (N T) c"]:
        B = latents.shape[0]
        # create temporal positions
        t_pos = centers(-1, 1, points_per_traj, device=latents.device)
        t_pos = einops.repeat(t_pos, "T -> B (N T) C", B=B, N=num_trajs, C=1)
        spatial_pos = einops.repeat(query_pos, "B N C -> B (N T) C", T=points_per_traj)  # B N*T 2
        pos = torch.cat([t_pos, spatial_pos], dim=-1)  # B ( n_emb + N*T ) 3
        return pos

    def decode_tracks(
        self,
        latents: Float[torch.Tensor, "B n_tokens C"],
        points_per_traj: int,
        num_trajs: int,
        latent_pos: Float[torch.Tensor, "B n_tokens 3"],
        query_pos: Float[torch.Tensor, "B N 2"],
        start_emb: Float[torch.Tensor, "B (H W) C"],
    ):
        B = latents.shape[0]
        latents = self.latents_in(latents)

        traj_pos = self.get_decode_pos(
            latents=latents,
            points_per_traj=points_per_traj,
            num_trajs=num_trajs,
            query_pos=query_pos,
        )
        # prepare decode tokens
        N_decode_tokens = num_trajs * points_per_traj
        decode_tokens = self.decode_token.expand(B, N_decode_tokens, -1)

        pos = torch.cat([latent_pos, traj_pos], dim=1)  # B ( n_emb + N*T ) c
        z = torch.cat([latents, decode_tokens], dim=1)

        x_cross: Float[torch.Tensor, "B (H W) C"] = start_emb  # [B (H W) C]
        pos_cross = self.pos_cross.expand(B, -1, -1)  # [B (H W) 3]
        decoded_tracks = self.backbone(z, pos, x_cross=x_cross, pos_cross=pos_cross)  # B (N T) C

        decoded_tracks = self.trajs_out(decoded_tracks[:, -N_decode_tokens:])
        decoded_tracks = einops.rearrange(decoded_tracks, "B (N T) C -> B N T C", N=num_trajs, T=points_per_traj)

        return decoded_tracks


class TrajEncoder(nn.Module):
    # annotate the buffers
    query_pos: Float[torch.Tensor, "1 n_latents 3"]
    reg_pos: Float[torch.Tensor, "1 1 3"]
    pos_cross: Float[torch.Tensor, "1 n_patches 3"]

    def __init__(
        self,
        d_model: int = 768,
        d_cross: int = 768,
        depth: int = 12,
        latent_dim: int = 16,
        grid_size: tuple[int, int, int] = (1, 16, 16),
        img_feat_size: tuple[int, int] = (16, 16),
        use_grid: bool = True,
    ):
        super().__init__()

        self.backbone = Level([TransformerLayer(d_model=d_model, d_cross=d_cross) for _ in range(depth)])
        self.grid_size = grid_size
        self.trajs_in = InputMLP(2, d_model, random_fourier=True)
        self.use_grid = use_grid

        self.reg_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.register_buffer("reg_pos", torch.zeros(1, 1, 3))  # [1, 1, 3]

        H, W = img_feat_size
        if use_grid:
            # for backward compatibility
            if isinstance(grid_size, int):
                grid_size = (1, grid_size, grid_size)

            self.query_token = nn.Parameter(torch.randn(1, 1, d_model))  # will be broadcasted in forward
            query_time = centers(-1, 1, grid_size[0])
            query_pos = torch.cat(
                # first is temporal pos, then spatial pos
                [
                    einops.repeat(
                        query_time,
                        "T -> (T H W) 1",
                        T=grid_size[0],
                        H=grid_size[1],
                        W=grid_size[2],
                    ),
                    einops.repeat(
                        make_axial_pos_2d(grid_size[-2], grid_size[-1]),
                        "(H W) C -> (T H W) C",
                        T=grid_size[0],
                        H=grid_size[1],
                        W=grid_size[2],
                    ),
                ],
                dim=-1,
            )  # [n_latents, 3]
        else:
            assert isinstance(grid_size, int), "use_grid is False only supported for int grid_size"
            print("Using non-grid latent tokens in TrajEncoder", flush=True)
            # use 0 positions for queries since we don't have a grid
            self.query_token = nn.Parameter(torch.randn(1, grid_size, d_model))
            query_pos = torch.zeros(grid_size, 3)  # [n_latents, 3]

        self.register_buffer("query_pos", query_pos[None])  # [1, n_latents, 3]

        # prepare cross attention pos encodings
        time_cross = -1.0
        pos_cross = torch.cat([torch.full((H * W, 1), time_cross), make_axial_pos_2d(H, W)], dim=-1)
        self.register_buffer("pos_cross", pos_cross)  # [H*W, 3]

        # VAE components
        self.to_mean = nn.Linear(d_model, d_model if latent_dim is None else latent_dim)
        self.to_logvar = nn.Linear(d_model, d_model if latent_dim is None else latent_dim)

    # tracks are in [-1, 1]
    def forward(
        self,
        tracks_yx: Float[torch.Tensor, "B N T 2"],
        start_emb: Float[torch.Tensor, "B (H W) C"],
    ) -> EncoderOutput:
        B, N, T, C = tracks_yx.shape

        start_pos_tracks = tracks_yx[:, :, 0, :]  # (B, N, 2)
        tracks_embedded = self.trajs_in(tracks_yx)

        # flatten tracks
        tracks_embedded = einops.rearrange(tracks_embedded, "B N T C -> B (N T) C")
        # query tokens arranged as grid
        if self.use_grid:
            n_grid_tokens = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
            query_tokens = self.query_token.expand(B, n_grid_tokens, -1)
        else:
            query_tokens = self.query_token.expand(B, -1, -1)
            n_grid_tokens = self.query_token.shape[1]
        reg_token = self.reg_token.expand(B, -1, -1)

        # create temporal positions
        t_pos = centers(-1, 1, T, device=tracks_yx.device)
        t_pos = einops.repeat(t_pos, "T -> B (N T) 1", B=B, N=N)
        token_pos = torch.cat([t_pos, einops.repeat(start_pos_tracks, "B N C -> B (N T) C", T=T)], dim=-1)
        query_pos = self.query_pos.expand(B, -1, -1)
        reg_pos = self.reg_pos.expand(B, -1, -1)

        tracks_embedded = torch.cat([query_tokens, tracks_embedded, reg_token], dim=1)  # B (G*G + N*T) C
        pos = torch.cat([query_pos, token_pos, reg_pos], dim=1)

        # prepare Cross Attention
        if start_emb is not None:
            x_cross: Float[torch.Tensor, "B (H W) C"] = start_emb  # [B (H W) C]
            pos_cross = self.pos_cross.expand(B, -1, -1)  # [B (H W) 3]
            latents = self.backbone(tracks_embedded, pos, x_cross=x_cross, pos_cross=pos_cross)  # B (N T) C
        else:
            latents = self.backbone(tracks_embedded, pos)

        # Extract only the grid tokens for VAE
        grid_latents: Float[torch.Tensor, "B n_tokens C"] = latents[:, :n_grid_tokens]

        # VAE: compute mean and logvar for each grid token
        mean = self.to_mean(grid_latents)  # [B, n_tokens, C]
        logvar = self.to_logvar(grid_latents)  # [B, n_tokens, C]

        return EncoderOutput(mean, logvar, query_pos)


class ZipMoVAE(nn.Module):
    def __init__(
        self,
        kl_weight: float = 1e-7,
        unlock_img_embedder: bool = False,
    ):
        super(ZipMoVAE, self).__init__()

        self.encoder = TrajEncoder()
        self.decoder = TrajRegressorDecoderMAE()
        self.img_embedder = MinDino(
            "dinov2_vitb14_reg",
            reshape=False,
            out="features",
            requires_grad=unlock_img_embedder,
        )
        self.kl_weight = kl_weight

    def _get_start_emb(
        self,
        start_frame: Float[torch.Tensor, "B H W C"] | None,
        start_emb: Float[torch.Tensor, "B (H W) C"] | None,
    ) -> Float[torch.Tensor, "B (H W) C"]:
        if start_emb is not None:
            return start_emb
        assert start_frame is not None, "Either start_frame or start_emb must be provided"
        return self.img_embedder(start_frame)

    def encode(
        self,
        tracks_yx: Float[torch.Tensor, "B N T 2"],
        start_frame: Float[torch.Tensor, "B H W C"] | None = None,
        start_emb: Float[torch.Tensor, "B (H W) C"] | None = None,
    ):
        start_emb = self._get_start_emb(start_frame=start_frame, start_emb=start_emb)

        enc_out: EncoderOutput = self.encoder(tracks_yx, start_emb=start_emb)
        return enc_out

    def decode(
        self,
        latents: Float[torch.Tensor, "B n_tokens C"],
        query_pos: Float[torch.Tensor, "B N 2"],
        points_per_track: int,
        start_frame: Float[torch.Tensor, "B H W C"] | None = None,
        start_emb: Float[torch.Tensor, "B (H W) C"] | None = None,
    ):
        B, N, _ = query_pos.shape
        latent_pos = self.encoder.query_pos.expand(B, -1, -1)

        start_emb = self._get_start_emb(start_frame=start_frame, start_emb=start_emb)

        pred_tracks: Float[torch.Tensor, "B N T 2"] = self.decoder.decode_tracks(
            latents,
            points_per_traj=points_per_track,
            num_trajs=N,
            query_pos=query_pos,
            latent_pos=latent_pos,
            start_emb=start_emb,
        )

        return pred_tracks

    def decode_dense(
        self,
        latents: Float[torch.Tensor, "B n_tokens C"],
        points_per_track: int,
        start_frame: Float[torch.Tensor, "B H W C"] | None = None,
        start_emb: Float[torch.Tensor, "B (H W) C"] | None = None,
        grid_size=64,
        chunk_size=8,
    ):
        B = latents.shape[0]
        dense_query_grid = make_axial_pos_2d(grid_size, grid_size, device=latents.device)  # [h*w, d]
        dense_query_grid = einops.repeat(dense_query_grid, "(h w) d -> b h w d", b=B, h=grid_size, w=grid_size, d=2)

        # Chunk into strided 8x8 grids
        # this is 64 tokens. our decoder was trained on 80
        # not perfect but should do the job
        chunks = chunk_grid_strided(
            dense_query_grid, chunk_size=chunk_size
        )  # (B, num_chunks, chunk_size, chunk_size, 2)
        num_chunks = chunks.shape[1]
        assert num_chunks * chunk_size * chunk_size == grid_size * grid_size, (
            f"Chunking error: {num_chunks} * {chunk_size} * {chunk_size} != {grid_size} * {grid_size}"
        )
        query_pos = einops.rearrange(chunks, "B num_chunks h w d -> (B num_chunks) (h w) d")  # (B, 64, 2)

        latents = einops.repeat(latents, "B n_tokens C -> (B num_chunks) n_tokens C", num_chunks=num_chunks)
        if start_frame is not None:
            start_frame = einops.repeat(start_frame, "B H W C -> (B num_chunks) H W C", num_chunks=num_chunks)
        if start_emb is not None:
            start_emb = einops.repeat(start_emb, "B n_emb C -> (B num_chunks) n_emb C", num_chunks=num_chunks)

        pred = self.decode(
            latents,
            query_pos=query_pos,
            points_per_track=points_per_track,
            start_frame=start_frame,
            start_emb=start_emb,
        )  # [B N T 2]

        chunked_pred = einops.rearrange(
            pred,
            "(B num_chunks) (h w) T c -> (B T) num_chunks h w c",
            B=B,
            h=chunk_size,
            w=chunk_size,
            c=2,
            T=points_per_track,
        )  # [B, num_chunks, h, w, T, 2]

        # Assemble back to full grid

        reconstructed_grid = assemble_chunks_strided(chunked_pred, grid_size)  # (B*T, grid_height, grid_width, 2)

        reconstructed_grid = einops.rearrange(
            reconstructed_grid, " (B T) H W c -> B H W T c", B=B, T=points_per_track
        )  # [B, H, W, T, 2]

        return reconstructed_grid

    def roundtrip(
        self,
        tracks_enc_yx: Float[torch.Tensor, "B N T 2"],
        start_frame: Float[torch.Tensor, "B H W C"],
        tracks_dec_yx: Float[torch.Tensor, "B N T 2"] | None,
        sample_latent: bool = True,
        decode_dense: bool = False,
        **decode_kwargs,
    ) -> tuple[
        Float[torch.Tensor, "B N T 2"],
        Float[torch.Tensor, "B n_tokens C"],
        Float[torch.Tensor, "B n_tokens 3"],
        Float[torch.Tensor, "B n_tokens C"],
        Float[torch.Tensor, "B n_tokens C"],
    ]:
        B, N, T, C = tracks_enc_yx.shape

        start_emb = self._get_start_emb(start_frame=start_frame, start_emb=None)

        enc_out: EncoderOutput = self.encoder(tracks_enc_yx, start_emb=start_emb)
        mean, logvar, latent_pos = enc_out.mean, enc_out.logvar, enc_out.latent_pos

        if sample_latent:
            latents = reparameterize(mean, logvar)
        else:
            latents = mean

        if decode_dense:
            pred_tracks: Float[torch.Tensor, "B N T 2"] = self.decode_dense(
                latents,
                points_per_track=T,
                start_emb=start_emb,
                **decode_kwargs,
            )
        else:
            assert tracks_dec_yx is not None, "tracks_dec_yx must be provided for non-dense decoding"
            start_pos = tracks_dec_yx[:, :, 0, :]  # (B, N, 2)
            pred_tracks: Float[torch.Tensor, "B N T 2"] = self.decode(
                latents,
                query_pos=start_pos,
                points_per_track=T,
                start_emb=start_emb,
            )

        return pred_tracks, latents, latent_pos, mean, logvar

    # N = number of trajectories
    # T = number of time steps (samples per trajectory)
    # compiled in init
    def forward(
        self,
        tracks_enc_yx: Float[torch.Tensor, "B N T 2"],
        tracks_dec_yx: Float[torch.Tensor, "B N T 2"],
        start_frame: Float[torch.Tensor, "B H W C"],
        **data_kwargs,
    ):
        start_emb = self._get_start_emb(start_frame=start_frame, start_emb=None)

        enc_out: EncoderOutput = self.encoder(tracks_enc_yx, start_emb=start_emb)
        mean, logvar, latent_pos = enc_out.mean, enc_out.logvar, enc_out.latent_pos

        latents = reparameterize(mean, logvar)
        std = torch.exp(0.5 * logvar)

        # Calculate reconstruction loss
        reconstruction_loss = self.decoder(latents, tracks_dec_yx, latent_pos=latent_pos, start_emb=start_emb)

        # Calculate KL divergence loss
        kl_loss = self.kl_weight * kl_divergence(mean, logvar).mean()

        # Combine losses
        loss_dict = dict(reconstruction_loss=reconstruction_loss, kl_loss=kl_loss)

        metrics = {
            "logvar_mean": logvar.mean(),
            "logvar_std": logvar.std(),
            "std_mean": std.mean(),
            "std_std": std.std(),
            "mean_mean": mean.mean(),
            "mean_std": mean.std(),
        }

        return loss_dict, metrics

    @torch.no_grad()
    def validate(
        self,
        dataloader_val,
        train_step: int,
        global_rank: int,
        max_steps: int | None,
        device,
    ) -> None:
        metrics_dict = defaultdict(list)

        for i, val_batch in enumerate(tqdm(dataloader_val, desc=f"Rank {global_rank} Validating", total=max_steps)):
            if max_steps is not None and i >= max_steps:
                break

            tracks_enc_yx = val_batch["tracks_enc_yx"].to(device=device)
            tracks_dec_yx = val_batch["tracks_dec_yx"].to(device=device)
            start_frame = val_batch["start_frame"].to(device=device)

            pred_tracks_yx, _, _, _, _ = self.roundtrip(
                tracks_enc_yx=tracks_enc_yx,
                start_frame=start_frame,
                tracks_dec_yx=tracks_dec_yx,
            )

            l1 = torch.nn.functional.l1_loss(pred_tracks_yx, tracks_dec_yx)
            l2 = torch.nn.functional.mse_loss(pred_tracks_yx, tracks_dec_yx)
            pck_metrics = calculate_pck(tracks_dec_yx, pred_tracks_yx)
            mean_pck = torch.mean(torch.stack(list(pck_metrics.values())))

            metrics_dict["Val/L1"].append(l1)
            metrics_dict["Val/L2"].append(l2)
            for threshold, pck_value in pck_metrics.items():
                metrics_dict[f"Val/PCK@{threshold}"].append(pck_value)
            metrics_dict["Val/PCK@Mean"].append(mean_pck)

        if not metrics_dict:
            return

        metric_names = sorted(metrics_dict.keys())
        avg_metrics = torch.stack([torch.stack(metrics_dict[name]).mean() for name in metric_names])

        if dist.is_initialized():
            dist.all_reduce(avg_metrics)
            avg_metrics = avg_metrics / dist.get_world_size()

        if global_rank == 0:
            wandb.log({name: avg_metrics[idx].item() for idx, name in enumerate(metric_names)}, step=train_step)


def calculate_pck(
    gt_yx: Float[torch.Tensor, "B N T 2"],
    pred_yx: Float[torch.Tensor, "B N T 2"],
    thresholds: list[float] = [1, 2, 4, 8, 16],  # this is relative to to the eval space resolution
    pck_eval_space: tuple[int, int] = (256, 256),  # this is the tap next resolution
) -> dict[str, torch.Tensor]:
    """
    Calculate Percentage of Correct Keypoints (PCK) for trajectory prediction.

    Args:
        gt_tracks: Ground truth tracks in normalized coordinates [-1, 1]
        pred_tracks: Predicted tracks in normalized coordinates [-1, 1]
        thresholds: List of distance thresholds for PCK calculation (in normalized coordinates)

    Returns:
        Dictionary mapping threshold to PCK value
    """

    # convert normalized coordinates [-1, 1] to pixel space [0, H-1] / [0, W-1]
    H, W = pck_eval_space
    scale = torch.tensor([H - 1, W - 1], device=gt_yx.device, dtype=gt_yx.dtype)
    gt_yx = (gt_yx + 1) / 2 * scale
    pred_yx = (pred_yx + 1) / 2 * scale

    # Calculate L2 distance between predicted and ground truth points
    distances = torch.norm(pred_yx - gt_yx, dim=-1)  # [B, N, T]

    pck_results = {}
    for threshold in thresholds:
        # Count points within threshold
        correct_points = (distances <= threshold).float()
        # Calculate percentage of correct points
        pck: torch.Tensor = correct_points.mean()
        if isinstance(threshold, float):
            pck_results[f"{threshold:.3f}"] = pck
        elif isinstance(threshold, int):
            pck_results[f"{threshold}px"] = pck
        else:
            raise ValueError("Threshold must be int or float")

    return pck_results
