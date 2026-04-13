import math
from collections.abc import Mapping

import torch
from einops import rearrange, repeat
from jaxtyping import Float, Int
from torch import nn
from functools import partial
from utils.libero_utils.viz import sample_grid
from zipmo.blocks import Level, SimpleProj, SimpleProjIn, TransformerLayer
from zipmo.planner import ZipMoPlanner_Libero
from zipmo.rope import make_axial_pos_2d
from zipmo.vae import ZipMoVAE


class PolicyHead(nn.Module):
    @staticmethod
    def checkpoint_uses_t_input(state_dict: Mapping[str, torch.Tensor]) -> bool:
        return any(
            key.startswith("track_predictor.start_t_mapping") or ".track_predictor.start_t_mapping" in key
            for key in state_dict
        )

    def __init__(
        self,
        temp_disc_fac: float = 0.99,
        temp_horizon: int = 16,
        track_pred_nfe: int = 10,
        vis_tracks: bool = False,  # visualize track predictions during rollout
        temporal_avg: bool = True,  # use temporal averaging of action predictions during rollout
        track_pred_cfg: float = 1.0,
        track_emb_dim: int = 16,
        model_dim: int = 768,
        depth: int = 6,
        track_predictor_use_t_input: bool = False,
        compile_track_predictor: bool = False,
        compile_vae_decode: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        track_ae = ZipMoVAE()
        self.track_predictor = ZipMoPlanner_Libero(vae=track_ae, use_t_input=track_predictor_use_t_input)

        self.vis_tracks = vis_tracks

        if compile_track_predictor:
            self.track_predictor.backbone.forward = torch.compile(
                self.track_predictor.backbone.forward,
                mode="default",
                dynamic=True,
            )

        if self.vis_tracks and compile_vae_decode:  # optional compile for faster rollout visualization
            self.track_predictor.vae.decode = torch.compile(
                self.track_predictor.vae.decode, mode="reduce-overhead", dynamic=True
            )

        self.track_predictor.cfg_scale = track_pred_cfg
        self.track_predictor.requires_grad_(False)

        self.policy_net = Level(
            [
                TransformerLayer(
                    d_model=model_dim,
                    d_cross=model_dim,
                    self_rope_mode="2d",
                    cross_rope_mode="2d",
                )
                for _ in range(depth)
            ]
        )

        self.policy_in_proj = SimpleProjIn(model_dim, model_dim)
        self.policy_out_proj = SimpleProj(model_dim, 7)  # action has 7 dims

        self.track_embedder = nn.Linear(track_emb_dim, model_dim)
        self.proprio_embedder = nn.Linear(9, model_dim)

        self.track_pred_nfe = track_pred_nfe
        self.temp_disc_fac = temp_disc_fac  # discount factor for future prediction errors
        self.temp_horizon = temp_horizon

        self.learned_action_queries = nn.Parameter(
            torch.randn(1, temp_horizon, model_dim) * 0.02, requires_grad=True
        )  # 1 T C

        self.action_queue = []
        self._rollout_generator = None

        # 6 modalities (we only use track predictions and proprioception): start frame view 1, start frame view 2, track pred view 1, track pred view 2, task_emb, proprio
        self.learnable_modality_emb = nn.Parameter(torch.randn(6, model_dim) * 0.02, requires_grad=True)

        self.extra_state_keys = [
            "joint_states",
            "gripper_states",
        ]  # required for compatibility with rollout func
        self.temporal_avg = temporal_avg

    def prepare_conds(
        self,
        start_frame: Float[torch.Tensor, "B V H W C"],
        track_pred: Float[torch.Tensor, "B (V nh nw) C"],
        proprio: Float[torch.Tensor, "B 9"],
    ):
        B, V, H, W, C = start_frame.shape

        track_emb = self.track_embedder(track_pred)  # B, (V * 16*16), C

        proprio_emb = self.proprio_embedder(proprio)

        # add learnable modality embeddings
        n_track_tokens = track_emb.shape[1] // V

        nh, nw = int(math.sqrt(n_track_tokens)), int(math.sqrt(n_track_tokens))
        track_emb = track_emb + repeat(
            self.learnable_modality_emb[2:4, :], "v c -> b (v n) c", b=B, n=nh * nw
        )  # track pred views

        proprio_emb = proprio_emb + repeat(self.learnable_modality_emb[5, :], "c -> b c", b=B)  # proprio

        # construct CA tokens and pos encodings
        pos_grid = make_axial_pos_2d(nh, nw, device=track_pred.device)  # (nh nw) 2
        cross_cond_list = []
        cross_pos_list = []

        cross_cond_list.append(track_emb)
        cross_pos_list += [pos_grid, pos_grid]  # 2 grids for 2 track views

        cross_cond_list.append(rearrange(proprio_emb, "b c -> b 1 c"))
        cross_pos_list.append(torch.zeros((1, 2), device=track_pred.device))  # 1 token for proprio

        cross_tokens = torch.cat(cross_cond_list, dim=1)  # B, n_cross , C
        pos_cross = torch.cat(cross_pos_list)

        return cross_tokens, repeat(pos_cross, "n c -> b n c", b=B)

    def reset(self, seed: int | None = None):
        self.action_queue = []
        self._rollout_generator = None
        if seed is not None:
            device = next(self.parameters()).device
            self._rollout_generator = torch.Generator(device=device)
            self._rollout_generator.manual_seed(int(seed))

    def forward(
        self,
        start_frame: Float[torch.Tensor, "B V H W C"],
        task_emb: Float[torch.Tensor, "B C"],
        actions: Float[torch.Tensor, "B T C"],
        joint_states: Float[torch.Tensor, "B T J"],
        gripper_states: Float[torch.Tensor, "B T G"],
        start_t: Int[torch.Tensor, "B"] | None = None,
        **kwargs,
    ):
        B, T, C = actions.shape
        with torch.no_grad():
            sample_tensor = torch.randn((B, *self.track_predictor.val_shape), device=actions.device)
            track_conds = torch.randn((B, 1, 5), device=actions.device)  # dummy track conditions
            view_id = torch.zeros((B,), dtype=torch.long, device=actions.device)  # dummy view id
            task_emb = torch.Tensor(task_emb).to(device=actions.device, dtype=actions.dtype)

            track_pred = self.track_predictor.sample(
                sample_tensor,
                start_frame=start_frame,
                txt_emb=rearrange(task_emb, "b c -> b 1 c"),
                decode_latent=False,
                track_conds=track_conds,
                view_id=view_id,
                sample_steps=self.track_pred_nfe,
                start_t=start_t,
            )  # B, (V * 16*16), C

        ### Get proprioception at current timestep (joint states and gripper states)
        joint_states = joint_states[:, 0, :]  # B, 7
        gripper_states = gripper_states[:, 0, :]  # B, 2
        proprio = torch.cat([joint_states, gripper_states], dim=-1).to(track_pred.dtype)  # B, 9

        # Prepare tokens for CA conditioning
        cross_tokens, pos_cross = self.prepare_conds(
            start_frame=start_frame,
            track_pred=track_pred,
            proprio=proprio,
        )
        cond = {"x_cross": cross_tokens, "pos_cross": pos_cross}

        pos = torch.zeros((B, T, 2), device=track_pred.device)  # learnable action tokens dont have spatial position

        action_qs_proj, pos_proj = self.policy_in_proj(repeat(self.learned_action_queries, "1 T C -> B T C", B=B), pos)
        actions_pred = self.policy_net(action_qs_proj, pos_proj, **cond)
        actions_pred = self.policy_out_proj(actions_pred)  # B T C

        # we discount prediction errors far in the future
        pred_err = (actions_pred - actions) ** 2  # B T C
        discount = self.temp_disc_fac ** torch.arange(T, device=pred_err.device)
        pred_err_discounted = pred_err * repeat(discount, "T -> B T C", B=B, C=pred_err.shape[2])  # B T C
        return pred_err_discounted.mean()

    @torch.no_grad()
    def act(self, obs, task_emb, extra_states):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device

        start_frame = torch.Tensor(obs).to(device=device, dtype=dtype)

        start_frame = start_frame / 127.5 - 1.0  # normalize to [-1, 1]

        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}

        proprio = torch.cat([extra_states["joint_states"], extra_states["gripper_states"]], dim=-1)  # B, 9

        B, V = start_frame.shape[:2]

        current_t = torch.tensor([len(self.action_queue)] * B, device=device)

        # GET TRACK PREDICTION
        with torch.autocast(device_type=start_frame.device.type, dtype=torch.bfloat16):
            sample_tensor = torch.randn(
                (B, *self.track_predictor.val_shape),
                device=device,
                generator=self._rollout_generator,
            )
            track_conds = torch.randn(
                (B, 1, 5),
                device=device,
                generator=self._rollout_generator,
            )  # dummy track conditions
            view_id = torch.zeros((B,), dtype=torch.long, device=device)  # dummy view id

            track_pred = self.track_predictor.sample(
                sample_tensor,
                start_frame=start_frame,
                txt_emb=rearrange(task_emb, "b c -> b 1 c"),
                decode_latent=False,
                track_conds=track_conds,
                view_id=view_id,
                sample_steps=self.track_pred_nfe,
                start_t=current_t,
            )  # B, (V * 16*16), C

        # Prepare tokens for CA conditioning
        cross_tokens, pos_cross = self.prepare_conds(
            start_frame=start_frame,
            track_pred=track_pred,
            proprio=proprio,
        )
        cond = {"x_cross": cross_tokens, "pos_cross": pos_cross}

        pos = torch.zeros(
            (B, self.learned_action_queries.shape[1], 2), device=track_pred.device
        )  # learnable action tokens dont have spatial position

        action_qs_proj, pos_proj = self.policy_in_proj(repeat(self.learned_action_queries, "1 T C -> B T C", B=B), pos)
        actions_pred = self.policy_net(action_qs_proj, pos_proj, **cond)
        actions_pred = self.policy_out_proj(actions_pred)  # B T C

        self.action_queue.append(actions_pred)

        current_t = torch.tensor([len(self.action_queue)] * B, device=device)

        current_horizon = min(len(self.action_queue), self.temp_horizon)

        if self.temporal_avg:
            # Compute discount weights in float64
            discount_weights_64 = self.temp_disc_fac ** torch.arange(
                current_horizon,
                device=track_pred.device,
                dtype=torch.float64,
            ).flip(0)

            discount_weights_64 = discount_weights_64 / discount_weights_64.sum()

            # Retrieve past predictions (still original dtype here)
            actions_pred_hist = torch.stack(self.action_queue[-current_horizon:], dim=1)  # B H T

            # Convert past actions to float64 for stable math
            actions_pred_hist_64 = actions_pred_hist.to(torch.float64)

            # Index out the correct action for each horizon step
            all_current_actions_64 = actions_pred_hist_64[
                :, torch.arange(current_horizon), -torch.arange(current_horizon)
            ]  # B H C (float64)

            # Expand discount weights to (B, H, C)
            discount_expanded_64 = discount_weights_64.view(1, current_horizon, 1).expand(
                actions_pred_hist_64.size(0),
                current_horizon,
                all_current_actions_64.size(-1),
            )

            # Weighted sum in float64
            current_action_64 = (discount_expanded_64 * all_current_actions_64).sum(dim=1)  # B C (float64)

            # Cast back to original dtype (bf16 or whatever)
            current_action = current_action_64.to(actions_pred.dtype)

        else:
            current_action = actions_pred[:, 0, :]  # B C

        if self.vis_tracks:
            # reshaped latent is b v t 1 n c
            latent = rearrange(track_pred, "b (v n) c -> (b v) n c", v=V)
            latent_denorm = self.track_predictor.denormalize_latents(
                latent
            )  # Unnormalize and un-center generated latents

            # we use an 8x8 grid for viz
            grid_points = sample_grid(
                8,
                device=track_pred.device,
                dtype=track_pred.dtype,
                left=(0.05, 0.05),
                right=(0.95, 0.95),
            )
            if grid_points.shape[0] < 80:
                # add random points if we have less than 80 points, to ensure consistent input size for decoder
                grid_points = torch.cat(
                    [
                        grid_points,
                        torch.rand(
                            (80 - grid_points.shape[0], 2),
                            device=track_pred.device,
                            dtype=track_pred.dtype,
                        ),
                    ],
                    dim=0,
                )

            query_pos = (repeat(grid_points, "n c -> b_new n c", b_new=B * V) - 0.5) * 2  # scale to [-1, 1]

            with torch.autocast(device_type=track_pred.device.type, dtype=torch.bfloat16):
                decoded_tracks = self.track_predictor.vae.decode(
                    latents=latent_denorm,
                    query_pos=query_pos,
                    points_per_track=64,
                    start_frame=rearrange(start_frame, "b v h w c -> (b v) h w c"),
                )

                decoded_tracks = (decoded_tracks + 1) / 2  # map back to [0, 1]

                vis_tracks = decoded_tracks[:, :, ::4, :].flip(-1)  # yx to xy, subsample from 64 to 16
                vis_tracks = vis_tracks[:, :64, :]  # only visualize first 64 points (8x8 grid) to avoid clutter
                vis_tracks = rearrange(vis_tracks, "(b v) n t_frames c -> b v t_frames n c", v=V, b=B)

                track_vis = (None, vis_tracks)

        else:
            track_vis = None

        return current_action.float().cpu().numpy(), track_vis


PolicyHeadATM = partial(PolicyHead, track_predictor_use_t_input=False)  # type: ignore
PolicyHeadTraMoE = partial(PolicyHead, track_predictor_use_t_input=True)  # type: ignore
