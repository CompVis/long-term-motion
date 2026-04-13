# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Stefan Baumann et al., CompVis @ LMU Munich
# Adapted from https://github.com/stefan-baumann/flow-poke-transformer/blob/main/scripts/demo/app.py by Nick Stracke

import glob
import math
import os
import tempfile
from dataclasses import dataclass, field
from functools import partial

import einops
import fire
import gradio as gr
import imageio
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import UInt8
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw
from torchvision.transforms.functional import resize
from torchvision.utils import flow_to_image

from zipmo.planner import ZipMoPlanner
from zipmo.rope import make_axial_pos_2d

VAL_SHAPE = [256, 16]


def draw_trajectories_on_frame(
    frame: torch.Tensor,  # [H, W, C] float in [0,1]
    tracks: torch.Tensor,  # [N, T, 2] pixel coords (x, y)
    start_cond: torch.Tensor | None = None,  # [N_cond, 3] (x, y, t)
    end_cond: torch.Tensor | None = None,  # [N_cond, 3] (x, y, t)
    traj_idx: torch.Tensor | None = None,  # [N_cond]
    line_width: float = 0.005,  # relative to image diagonal
    marker_size: float = 0.01,  # relative to image diagonal
    cond_size: float = 0.015,  # relative to image diagonal
    return_pil_image: bool = False,
    draw_pokes: bool = False,
    draw_trajectories: bool = True,
    poke_scale: float = 0.05,  # relative to image diagonal
    outline_width: float = 0.001,  # white edge width, relative to diagonal
    dpi: int = 100,
):
    """
    Draw trajectories with gradient lines and optional arrows using matplotlib.
    All sizes are relative to image diagonal for consistent appearance across resolutions.
    """
    assert frame.ndim == 3 and frame.shape[-1] == 3, "frame must be [H,W,3] RGB in [0,1]"
    H, W, _ = frame.shape

    # Scale all sizes relative to image diagonal
    # scale_factor = np.sqrt(H**2 + W**2)
    scale_factor = min(H, W) * 1.5
    line_px = math.ceil(line_width * scale_factor)
    marker_px = math.ceil(marker_size * scale_factor)
    cond_size_px = math.ceil(cond_size * scale_factor)
    poke_scale_px = math.ceil(poke_scale * scale_factor)
    outline_px = math.ceil(outline_width * scale_factor)
    # Convert to numpy
    frame_np = frame.float().clamp(0, 1).cpu().numpy()
    tracks_np = tracks.float().cpu().numpy()  # [N, T, 2]
    N, T, _ = tracks_np.shape

    # Create figure
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(frame_np, extent=(0, W, H, 0))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    cmap = cm.get_cmap("jet")

    # Draw each trajectory
    for n in range(N):
        pts = tracks_np[n]  # [T, 2]
        valid = ~np.isnan(pts).any(axis=1)
        pts_valid = pts[valid]

        if not valid.any() or len(pts_valid) < 2:
            continue

        start_marker_color = cmap(0.0)
        end_marker_color = cmap(1.0)
        start_end_dist = np.linalg.norm(pts_valid[-1] - pts_valid[0])
        if draw_pokes:  # less than 0.5% of movement
            start_marker_color = "black"
            end_marker_color = "black"

            if start_end_dist > scale_factor * 0.005:
                start = pts_valid[0]
                end = pts_valid[-1]
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="simple,head_length=0.8,head_width=0.6",
                    facecolor="black",
                    edgecolor="white",
                    linewidth=outline_px,
                    mutation_scale=poke_scale_px,
                    shrinkA=0,
                    shrinkB=0,
                    zorder=10,
                )
                ax.add_patch(arrow)

        start_pos = pts_valid[0]
        start_circle = Circle(
            start_pos,
            marker_px,
            facecolor=start_marker_color,
            edgecolor="white",
            linewidth=outline_px,
            zorder=4,
        )
        ax.add_patch(start_circle)

        end_pos = pts_valid[-1]
        end_circle = Circle(
            end_pos,
            marker_px,
            facecolor=end_marker_color,
            edgecolor="white",
            linewidth=outline_px,
            zorder=4,
        )
        ax.add_patch(end_circle)

        if not draw_trajectories:
            continue

        # Draw gradient trajectory lines
        for t in range(len(pts_valid) - 1):
            x1, y1 = pts_valid[t]
            x2, y2 = pts_valid[t + 1]
            ratio = t / max(len(pts_valid) - 1, 1)
            color = cmap(ratio)

            # White halo
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white",
                linewidth=line_px + 2,
                alpha=0.6,
                solid_capstyle="round",
                zorder=2,
            )
            # Colored line
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=line_px, solid_capstyle="round", zorder=3)

    # Draw condition markers if provided
    if start_cond is not None and end_cond is not None and traj_idx is not None:
        sc = start_cond.float().cpu().numpy()
        ec = end_cond.float().cpu().numpy()

        for i in range(len(sc)):
            sx, sy = sc[i, :2]
            ex, ey = ec[i, :2]

            # Start condition (cyan square)
            start_rect = Rectangle(
                (sx - cond_size_px / 2, sy - cond_size_px / 2),
                cond_size_px,
                cond_size_px,
                facecolor="none",
                edgecolor="cyan",
                linewidth=2,
                zorder=6,
            )
            ax.add_patch(start_rect)

            # End condition (magenta square)
            end_rect = Rectangle(
                (ex - cond_size_px / 2, ey - cond_size_px / 2),
                cond_size_px,
                cond_size_px,
                facecolor="none",
                edgecolor="magenta",
                linewidth=2,
                zorder=6,
            )
            ax.add_patch(end_rect)

    # Convert to image
    fig.canvas.draw()
    img_array = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if return_pil_image:
        from PIL import Image

        return Image.fromarray(img_array)

    return img_array


@dataclass
class ModelState:
    model: ZipMoPlanner
    device: torch.device


@dataclass
class SampleState:
    image: UInt8[np.ndarray, "h w c"]
    all_tracks: list[list[tuple[float, float]]] = field(default_factory=lambda: [[]])

    def add_point_ui(self, point: tuple[float, float]):
        self.all_tracks[-1].append(point)


def render_input(inputs: SampleState) -> UInt8[np.ndarray, "h w c"]:
    H, W, _ = inputs.image.shape
    dpi = 100
    px_scale = math.ceil(min(H, W) * 0.015)

    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(inputs.image, origin="upper", extent=(0, W, 0, H), aspect="equal")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    for track in inputs.all_tracks:
        if len(track) == 0:
            continue
        track_px = [(x * W, H - y * H) for (x, y) in track]  # we need to vertically flip bc of ax.imshow coord system

        if len(track) == 1:
            ax.scatter(track_px[0][0], track_px[0][1], color="C1", marker="x", s=150, linewidths=3)
        elif len(track_px) == 2 and np.linalg.norm(np.array(track_px[0]) - np.array(track_px[1]), ord=2) <= 0.01:
            # If there is ~no movement, draw a circle to prevent an invisible arrow
            arrow = Circle(xy=track_px[0], radius=px_scale, facecolor="black", edgecolor="white")
            ax.add_patch(arrow)
        else:
            for start, end in zip(track_px[:-1], track_px[1:]):
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="simple,head_length=0.8,head_width=0.6",
                    facecolor="black",
                    edgecolor="white",
                    linewidth=1,
                    mutation_scale=20,
                    shrinkA=0,
                    shrinkB=0,
                )
                ax.add_patch(arrow)

    fig.canvas.draw()
    img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def blend_image_with_white(
    image: UInt8[np.ndarray, "H W C"],
    white_mix: float = 0.45,
) -> UInt8[np.ndarray, "H W C"]:
    image_f = image.astype(np.float32) / 255.0
    blended = image_f * (1.0 - white_mix) + white_mix
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)


def sample_track_colors(
    image: UInt8[np.ndarray, "H W C"],
    tracks: torch.Tensor,
) -> list[tuple[int, int, int]]:
    h, w, _ = image.shape
    tracks_np = tracks.float().cpu().numpy()
    colors = []
    for pts in tracks_np:
        valid = ~np.isnan(pts).any(axis=1)
        if valid.any():
            start_x, start_y = pts[valid][0]
        else:
            start_x, start_y = 0.0, 0.0
        x = int(np.clip(np.rint(start_x), 0, w - 1))
        y = int(np.clip(np.rint(start_y), 0, h - 1))
        colors.append(tuple(int(channel) for channel in image[y, x]))
    return colors


def render_track_video(
    image: UInt8[np.ndarray, "H W C"],
    tracks: torch.Tensor,
    tail_length: int = 8,
    line_width: float = 0.007,
    marker_size: float = 0.012,
    initial_hold_frames: int = 8,
    fps: int = 12,
    white_mix: float = 0.45,
    motion_threshold: float = 1,  # in pixels
) -> tuple[list[UInt8[np.ndarray, "H W C"]], int]:
    assert image.ndim == 3 and image.shape[-1] == 3, "image must be [H, W, 3]"

    h, w, _ = image.shape
    scale_factor = min(h, w) * 1.5
    line_px = max(1, math.ceil(line_width * scale_factor))
    marker_px = max(2, math.ceil(marker_size * scale_factor))

    base_image = Image.fromarray(blend_image_with_white(image, white_mix=white_mix)).convert("RGBA")
    tracks_np = tracks.float().cpu().numpy()
    num_tracks, num_steps, _ = tracks_np.shape
    track_colors = sample_track_colors(image, tracks)
    frames: list[UInt8[np.ndarray, "H W C"]] = []
    moving_track_mask = np.zeros(num_tracks, dtype=bool)

    for track_idx, pts in enumerate(tracks_np):
        valid = ~np.isnan(pts).any(axis=1)
        pts_valid = pts[valid]
        if len(pts_valid) < 2:
            continue
        step_lengths = np.linalg.norm(np.diff(pts_valid, axis=0), axis=-1)
        mean_movement = step_lengths.mean()
        moving_track_mask[track_idx] = mean_movement >= motion_threshold

    def append_frame(draw_motion_until: int | None):
        canvas = base_image.copy()
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        for track_idx in range(num_tracks):
            if not moving_track_mask[track_idx]:
                continue
            pts = tracks_np[track_idx]
            color = track_colors[track_idx]
            valid = ~np.isnan(pts).any(axis=1)
            if not valid.any():
                continue

            pts_valid = pts[valid]
            if draw_motion_until is None:
                start_x, start_y = pts_valid[0]
                draw.ellipse(
                    (
                        start_x - marker_px,
                        start_y - marker_px,
                        start_x + marker_px,
                        start_y + marker_px,
                    ),
                    fill=(*color, 255),
                )
                continue

            if len(pts_valid) < 2:
                continue

            head_idx = min(draw_motion_until, len(pts_valid) - 1)
            tail_start = max(0, head_idx - tail_length + 1)

            for step in range(tail_start, head_idx):
                x1, y1 = pts_valid[step]
                x2, y2 = pts_valid[step + 1]
                alpha = int(255 * ((step - tail_start + 1) / max(head_idx - tail_start + 1, 1)))
                draw.line((x1, y1, x2, y2), fill=(*color, alpha), width=line_px)

            head_x, head_y = pts_valid[head_idx]
            draw.ellipse(
                (
                    head_x - marker_px,
                    head_y - marker_px,
                    head_x + marker_px,
                    head_y + marker_px,
                ),
                fill=(*color, 255),
            )

        frame = np.array(Image.alpha_composite(canvas, overlay).convert("RGB"))
        frames.append(frame)

    for _ in range(initial_hold_frames):
        append_frame(draw_motion_until=None)
    for step_idx in range(1, num_steps):
        append_frame(draw_motion_until=step_idx)

    return frames, fps


@torch.no_grad()
def predict(
    inputs: SampleState,
    model_state: ModelState,
    cfg_scale: float = 1.0,
    n_vis_tracks: int = 80,
    n_decoder_tracks: int = 80,
    decode_grid: bool = False,
    decode_dense: bool = False,
    seed: int = 42,
    render_videos: bool = False,
    motion_threshold: float = 1.0,
) -> tuple[UInt8[np.ndarray, "h w c"], None]:
    assert inputs.image is not None, "Image input is required for prediction."

    batch_size = 4

    # print(inputs, flush=True)
    device = model_state.device
    dtype = torch.bfloat16
    model = model_state.model
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    flatteneted_pokes = []
    for track in inputs.all_tracks:
        n_end = len(track) - 1

        if n_end < 1:
            print("Skipping track with less than 2 points:", track, flush=True)
            continue

        start = [c * 2 - 1 for c in track[0]]
        for i, end in enumerate(track[1:]):
            end = [c * 2 - 1 for c in end]
            flatteneted_pokes.append([start[1], start[0], end[1], end[0], (i + 1) / n_end])  # end_t in [0, 1]

    if len(flatteneted_pokes) > 0:
        track_conds = torch.tensor(flatteneted_pokes, device=device, dtype=dtype)  # (n_pokes, 5)
    else:
        track_conds = torch.zeros((0, 5), device=device, dtype=dtype)  # (n_pokes, 5)

    query_points = []
    for track in inputs.all_tracks:
        if len(track) > 0:
            start = track[0]
            query_points.append([start[1], start[0]])  # [0, 1] range
    query_points = torch.tensor(query_points, device=device, dtype=dtype)  # (n_query, 2)
    query_points = torch.cat(
        [
            query_points,
            torch.rand(n_decoder_tracks - query_points.shape[0], 2, device=device, dtype=dtype, generator=g),
        ],
        dim=0,
    )

    with torch.autocast(device_type="cuda", dtype=dtype):
        model.cfg_scale = cfg_scale
        pred_tracks = model.sample(
            z=torch.randn(batch_size, *VAL_SHAPE, device=device, dtype=dtype, generator=g),
            points_per_traj=64,
            query_pos=query_points.expand(batch_size, -1, -1) * 2 - 1,
            start_frame=torch.from_numpy(inputs.image).to(device).expand(batch_size, -1, -1, -1).float() / 127.5 - 1.0,
            track_conds=track_conds.expand(batch_size, -1, -1),  # (batch_size, n_pokes, 5)
            decode_dense=decode_dense,
        )

    H, W, C = inputs.image.shape

    pred_tracks = (pred_tracks.flip(-1) + 1) / 2 * torch.tensor([W, H], device=device)
    if decode_dense:
        pred_tracks = einops.rearrange(pred_tracks, "b h w t c -> b (h w) t c")

    video_paths = []
    out_frames = []

    for pred_track in pred_tracks:
        out_frame = draw_trajectories_on_frame(
            torch.from_numpy(inputs.image) / 255.0,
            pred_track[:n_vis_tracks].cpu(),
        )
        out_frames.append(out_frame)

        if render_videos:
            track_video_frames, track_video_fps = render_track_video(
                inputs.image,
                pred_track[:n_vis_tracks].cpu(),
                motion_threshold=motion_threshold,
            )
            tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_video.close()
            imageio.mimwrite(tmp_video.name, track_video_frames, format="mp4", fps=track_video_fps)
            video_paths.append(tmp_video.name)

    if render_videos and decode_grid:
        H = W = int(math.sqrt(n_decoder_tracks))

        batch_size = 1
        grid = make_axial_pos_2d(H, W, device=device, dtype=dtype)

        with torch.autocast(device_type="cuda", dtype=dtype):
            pred_tracks = model.sample(
                z=torch.randn(batch_size, *VAL_SHAPE, device=device, dtype=dtype),
                points_per_traj=64,
                query_pos=grid.expand(batch_size, -1, -1),
                start_frame=torch.from_numpy(inputs.image).to(device).expand(batch_size, -1, -1, -1).float() / 127.5
                - 1.0,
                track_conds=track_conds.expand(batch_size, -1, -1),  # (batch_size, n_pokes, 5)
                decode_dense=decode_dense,
            )
        if decode_dense:
            grid = einops.rearrange(pred_tracks[0], "h w t c -> t c h w")
        else:
            grid = einops.rearrange(pred_tracks, "b (h w) t c -> b t c h w", h=H, w=W)[0]
        flow = torch.diff(grid, dim=0)
        flow_video = flow_to_image(flow.float())
        flow_video = resize(flow_video, size=[256, 256])

        tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        imageio.mimwrite(tmpfile.name, flow_video.movedim(1, -1).cpu().numpy(), format="mp4", fps=15)

        video_paths = [tmpfile.name]

    # Ensure stable number of file outputs in the UI.
    video_paths = (video_paths + [None] * 4)[:4]
    return [np.concatenate(out_frames, axis=1), *video_paths]


def load_video(
    selected_filename: str | None,
    inputs: SampleState,
    video_dir: str,
):
    """Load first frame of a selected video and return (frame, SampleState)."""
    if selected_filename is None:
        return None, inputs
    full_path = os.path.join(video_dir, selected_filename)
    try:
        reader = imageio.get_reader(full_path)
        frame = reader.get_data(0)
    except Exception as e:
        print(f"Failed to read video {full_path}: {e}", flush=True)
        return None, inputs

    # Normalize frame to RGB uint8
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[2] == 4:
        frame = frame[:, :, :3]

    H, W, C = frame.shape
    target_size = 256
    min_size = min(H, W)

    scale = target_size / min_size
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    frame = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.BICUBIC))

    inputs = SampleState(image=frame)
    return frame, inputs


def demo(
    device: str = "cuda",
    compile: bool = False,
    share: bool = False,
    server_name: str = "0.0.0.0",
    server_port: int = 55555,
    video_pexel_dir="./data/pexels",
):
    with gr.Blocks() as demo:
        gr.Markdown("## Motion Spaces Demo")

        model: ZipMoPlanner = torch.hub.load("CompVis/long-term-motion", "zipmo_planner_sparse")  # type: ignore
        model.eval()
        model.to(device)
        model.requires_grad_(False)

        if compile:
            model._predict_velocity = torch.compile(model._predict_velocity)

        gr_model = gr.State(ModelState(model=model, device=torch.device(device)))

        with gr.Row():
            # Input & poke/query interaction handling
            with gr.Column():
                inputs = gr.State(None)
                image_input = gr.Image(
                    label="Input Image", interactive=True, type="numpy", image_mode="RGB", format="png"
                )

                new_track_button = gr.Button("Start New Track")
                new_track_button.click(
                    lambda inputs: inputs.all_tracks.append([]) or inputs,
                    inputs=[inputs],
                    outputs=[inputs],
                )
                reset_tracks_button = gr.Button("Reset Tracks")
                reset_tracks_button.click(
                    lambda inputs: setattr(inputs, "all_tracks", [[]]) or inputs,
                    inputs=[inputs],
                    outputs=[inputs],
                )

                # Directory with Pexel videos to choose from
                pexel_video_paths = sorted(
                    sum(
                        (
                            glob.glob(os.path.join(video_pexel_dir, f"original*.{ext}"))
                            for ext in ["mp4", "avi", "mov", "mkv"]
                        ),
                        [],
                    )
                )
                pexel_video_choices = [os.path.basename(p) for p in pexel_video_paths]
                with gr.Row():  # dropdown and load button side-by-side
                    video_pexel_dropdown = gr.Dropdown(
                        choices=pexel_video_choices,
                        label="Select Pexels Video",
                        value=pexel_video_choices[0] if pexel_video_choices else None,
                    )
                    load_pexel_video_button = gr.Button("Load video")

                # register module-level load_video, binding demo's video_dir via partial
                load_pexel_video_button.click(
                    partial(load_video, video_dir=video_pexel_dir),
                    inputs=[video_pexel_dropdown, inputs],
                    outputs=[image_input, inputs],
                )

                @torch.no_grad()
                def preprocess_image(
                    image_input: UInt8[np.ndarray, "h w c"] | None,
                    inputs: SampleState,
                ):
                    if image_input is None:
                        return None, None
                    H, W, C = image_input.shape
                    assert C == 3, f"Expected 3 channels in the image, got {C} channels."
                    target_size: int = 256
                    min_size = min(H, W)

                    scale = target_size / min_size
                    new_h, new_w = int(round(H * scale)), int(round(W * scale))
                    frame = np.array(Image.fromarray(image_input).resize((new_w, new_h), Image.BICUBIC))

                    inputs = SampleState(
                        image=frame,
                    )
                    return frame, inputs

                image_input.upload(preprocess_image, inputs=[image_input, inputs], outputs=[image_input, inputs])

                def input_click(image_input, inputs: SampleState, evt: gr.SelectData):
                    if image_input is None:
                        return image_input, inputs
                    # (x, y) indexing
                    inputs.add_point_ui((evt.index[0] / image_input.shape[1], evt.index[1] / image_input.shape[0]))
                    return render_input(inputs), inputs

                image_input.select(input_click, inputs=[image_input, inputs], outputs=[image_input, inputs])

            # Output visualization
            with gr.Column():
                image_output = gr.Image(label="Prediction", type="numpy", image_mode="RGB", format="png")

                with gr.Row():
                    video_outputs_sep = [gr.Video(label=f"Prediction {i + 1}", format="mp4") for i in range(4)]

                predict_button = gr.Button("Predict", variant="primary")

                with gr.Accordion("Advanced Settings"):
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=7, step=0.1, value=1)
                    n_decoder_tracks = gr.Number(label="Number of Decoder Tracks", value=80, precision=0, minimum=1)
                    n_vis_tracks = gr.Number(label="Number of Visualization Tracks", value=80, precision=0, minimum=1)
                    render_videos = gr.Checkbox(label="Render Video Visualization", value=False)
                    decode_grid = gr.Checkbox(label="Use Decoding Grid", value=False)
                    decode_dense = gr.Checkbox(label="Use Dense Decoding", value=False)
                    seed = gr.Slider(label="Random Seed", minimum=0, maximum=2**32 - 1, step=1, value=42)
                    motion_threshold = gr.Slider(
                        label="Motion Threshold", minimum=0.0, maximum=5.0, step=0.1, value=1.0
                    )

                predict_button.click(
                    predict,
                    inputs=[
                        inputs,
                        gr_model,
                        cfg_scale,
                        n_vis_tracks,
                        n_decoder_tracks,
                        decode_grid,
                        decode_dense,
                        seed,
                        render_videos,
                        motion_threshold,
                    ],
                    outputs=[
                        image_output,
                        *video_outputs_sep,
                    ],
                    show_progress=True,
                )

    print(f"demo launched at {server_name}:{server_port}", flush=True)
    demo.launch(share=share, server_name=server_name, server_port=server_port, debug=True)


if __name__ == "__main__":
    # Allow TF32, make model go brrrrrrrrrr
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable benchmarking for matmuls
    torch.backends.cudnn.benchmark = True
    # Increase the compilation cache size, so that we don't stop compiling when hitting many different paths
    torch._dynamo.config.cache_size_limit = max(2**12, torch._dynamo.config.cache_size_limit)

    # This seems to be crucial for compilation with cudagraphs to be stable (as of torch ~2.8) with gradio's multithreading
    # If you do inference using a normal script, leave these enabled, as it'll likely give you slight speedups
    # Also, if you run out of memory, consider enabling these again; as it might help with that, although you might pay
    # with the aforementioned stability issues
    torch._inductor.config.triton.cudagraph_trees = False

    fire.Fire(demo)
