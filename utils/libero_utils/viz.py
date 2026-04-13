# Adapted from https://github.com/Large-Trajectory-Model/ATM

from einops import rearrange, repeat
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def sample_grid(n, device="cuda", dtype=torch.float32, left=(0.1, 0.1), right=(0.9, 0.9)):
    # sample nxn points as a grid
    u = torch.linspace(left[0], right[0], n, device=device, dtype=dtype)
    v = torch.linspace(left[1], right[1], n, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v)
    u = u.reshape(-1)
    v = v.reshape(-1)
    points = torch.stack([u, v], dim=-1)
    return points


def make_grid_video_from_numpy(video_array, ncol, output_name="./output.mp4", speedup=1, padding=5, **kwargs):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=padding)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)


def save_numpy_as_video(array, filename, fps=20, extension="mp4"):
    frames = np.asarray(array)
    if np.max(frames) <= 2.0:
        frames = frames * 255.0
    frames = np.clip(frames, 0, 255).astype(np.uint8)

    if frames.ndim == 3:
        frames = np.repeat(frames[..., np.newaxis], 3, axis=-1)
    elif frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    output_path = Path(filename).with_suffix(f".{extension}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames.shape[1], frames.shape[2]
    ext = output_path.suffix.lower()
    fourcc_str = "mp4v" if ext in {".mp4", ".mov"} else "XVID"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*fourcc_str),
        float(fps),
        (width, height),
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    return str(output_path)


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid(array, ncol=5, padding=0, pad_value=120):
    """numpy version of the make_grid function in torch. Dimension of array: NHWC"""
    if np.max(array) < 2.0:
        array = array * 255.0
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    if N % ncol > 0:
        res = ncol - N % ncol
        array = np.concatenate([array, np.ones([res, H, W, C])])
        N = array.shape[0]
    nrow = N // ncol
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(
            array[idx],
            [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]],
            constant_values=pad_value,
            mode="constant",
        )
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(
                array[idx],
                [[padding if i == 0 else 0, padding], [0, padding], [0, 0]],
                constant_values=pad_value,
                mode="constant",
            )
            row = np.hstack([row, cur_img])
        idx += 1
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img.astype(np.float32)


def combine_track_and_img(track: torch.Tensor, vid: np.ndarray, alpha: float = 1.0):
    """
    track: [B, T, N, 2]
    vid: [B, C, H, W]
    alpha: track overlay opacity in [0, 1]
    return: (B, C, H, W)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    img_size = vid.shape[-1]
    track_video = tracks_to_video(track, img_size)  # B 3 H W
    track_video = track_video.detach().cpu().numpy()
    vid = vid.copy().astype(np.float32)
    if alpha == 0.0:
        return vid.astype(np.uint8)

    mask = track_video > 0
    if alpha < 1.0:
        vid[mask] = (1.0 - alpha) * vid[mask] + alpha * track_video[mask]
    else:
        vid[mask] = track_video[mask]
    return vid.astype(np.uint8)


def tracks_to_video(tracks, img_size, alpha: float = 1.0):
    """
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    alpha: track intensity scale in [0, 1]
    return: (B, C, H, W)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    B, T, N, _ = tracks.shape
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # b, t, c, h, w
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]

    # Get blue to purple cmap
    cmap = plt.get_cmap("coolwarm")
    cmap = cmap(1 / np.arange(T))[:T, :3][::-1]
    binary_vid = binary_vid.clone()

    for l in range(T):
        # interpolate betweeen blue and red
        binary_vid[:, l, 0] = binary_vid[:, l, 0] * cmap[l, 0] * 255 * alpha
        binary_vid[:, l, 1] = binary_vid[:, l, 1] * cmap[l, 1] * 255 * alpha
        binary_vid[:, l, 2] = binary_vid[:, l, 2] * cmap[l, 2] * 255 * alpha
    # Overwride from the last frame
    track_vid = torch.sum(binary_vid, dim=1)
    track_vid[track_vid > 255] = 255
    return track_vid


def tracks_to_binary_img(tracks, img_size):
    """
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    return: (B, T, C, H, W)
    """

    B, T, N, C = tracks.shape
    generation_size = 128
    H, W = generation_size, generation_size

    tracks = tracks * generation_size
    u, v = tracks[:, :, :, 0].long(), tracks[:, :, :, 1].long()
    u = torch.clamp(u, 0, W - 1)
    v = torch.clamp(v, 0, H - 1)
    uv = u + v * W

    img = torch.zeros(B, T, H * W).to(tracks.device)
    img = img.scatter(2, uv, 1).view(B, T, H, W)

    # img size is b x t x h x w
    img = repeat(img, "b t h w -> (b t) h w")[:, None, :, :]

    # Generate 5x5 gaussian kernel
    kernel = [
        [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
        [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
        [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
        [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
        [0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
    ]
    kernel /= np.max(kernel)
    kernel = torch.FloatTensor(kernel)[None, None, :, :].to(tracks.device)
    img = F.conv2d(img, kernel, padding=2)[:, 0, :, :]
    img = rearrange(img, "(b t) h w -> b t h w", b=B)
    if generation_size != img_size:
        img = F.interpolate(img, size=(img_size, img_size), mode="bicubic")
    img = torch.clamp(img, 0, 1)
    img = torch.where(img < 0.05, torch.tensor(0.0), img)

    img = repeat(img, "b t h w -> b t c h w", c=3)

    assert torch.max(img) <= 1
    return img


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame
