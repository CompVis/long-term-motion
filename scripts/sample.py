from pathlib import Path
from typing import Literal

import fire
import torch
import torchvision
from jaxtyping import Float
from PIL import Image
from tqdm.auto import tqdm

from scripts.demo import draw_trajectories_on_frame
from zipmo.planner import ZipMoPlanner


def get_track_cond_eval(tracks: Float[torch.Tensor, "B N T 2"]) -> Float[torch.Tensor, "B N_cond 5"]:
    start_points = tracks[:, :, 0]  # (B, N_cond, 2)
    end_points = tracks[:, :, -1]  # (B, N_cond, 2)

    combined_conds = torch.cat(
        [start_points, end_points, torch.full_like(end_points[:, :, :1], 1.0)], dim=-1
    )  # B, N_cond, 5
    return combined_conds.contiguous()


def concatenate_images_horizontally(images: list[Image.Image]) -> Image.Image:
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return new_im


def main(
    gt_path="./data/gt_tracks.pt",
    samples_path=Path("./data/pexels"),
    output_path=Path("./outputs/evals"),
    mode: Literal["sparse", "dense"] = "sparse",
    cfg_scale: float = 1.0,
    seed: int = 43,
    noviz: bool = False,
):
    samples_path = Path(samples_path)
    output_path = Path(output_path)

    data_dict = torch.load(
        str(gt_path),
        weights_only=False,
        map_location="cpu",
    )

    if mode == "sparse":
        output_path = output_path / f"sparse-cfg{cfg_scale}-seed{seed}"
        model: ZipMoPlanner = torch.hub.load("CompVis/long-term-motion", "zipmo_planner_sparse")  # type: ignore
        K = 8
    elif mode == "dense":
        output_path = output_path / f"dense-cfg{cfg_scale}-seed{seed}"
        model: ZipMoPlanner = torch.hub.load("CompVis/long-term-motion", "zipmo_planner_dense")  # type: ignore
        K = 128
        noviz = True  # too many samples to visualize

    model.eval()
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.cfg_scale = cfg_scale

    output_path.mkdir(parents=True, exist_ok=True)
    results = {}

    for video_name, data in tqdm(data_dict.items()):
        gt_tracks = data["tracks"]  # [40, 64, 2] in [-1, 1]
        gt_queries = gt_tracks[:, 0, :].clone()  # [40, 2] in [-1, 1]
        n_gt_tracks = gt_tracks.shape[0]
        assert n_gt_tracks == 40, f"Expected 40 GT tracks, but got {n_gt_tracks} for video {video_name}"

        video_path = samples_path / f"original-{video_name}.mp4"
        video, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec")  # (T, H, W, C) in [0, 255]
        start_frame = video[0].float() / 255.0  # [H, W, C] in [0, 1]
        start_frame = start_frame * 2 - 1  # in [-1, 1]
        H, W = start_frame.shape[:2]

        results[video_name] = {
            "gt": {
                "tracks": gt_tracks.cpu().float(),
                "queries": gt_queries.cpu().float(),
            }
        }

        cond_tracks = gt_tracks.to(device="cuda", dtype=torch.bfloat16).expand(K, -1, -1, -1)  # [B, 40, T, 2]
        cond_tracks = get_track_cond_eval(cond_tracks)  # [B, 40, 5]

        viz_tracks = gt_tracks.mul(0.5).add(0.5) * torch.tensor([H - 1, W - 1], device=gt_tracks.device)[None, None, :]
        gt_viz = draw_trajectories_on_frame(
            start_frame * 0.5 + 0.5,
            viz_tracks.flip(-1).cpu().float(),
            return_pil_image=True,
        )

        if mode == "dense":
            poke_list = [40]
        else:
            poke_list = [1, 2, 4, 8, 16]

        for n_pokes in poke_list:
            ours_name = f"zipmo-{n_pokes}_pokes"
            results[video_name][ours_name] = {
                "tracks": [],
            }
            generator = torch.Generator(device="cuda").manual_seed(seed)
            z = torch.randn(K, 16 * 16, 16, device="cuda", dtype=torch.bfloat16, generator=generator)

            viz_poke_tracks = (
                gt_tracks[:n_pokes].mul(0.5).add(0.5)
                * torch.tensor([H - 1, W - 1], device=gt_tracks.device)[None, None, :]
            )
            poke_viz = draw_trajectories_on_frame(
                start_frame * 0.5 + 0.5,
                viz_poke_tracks.flip(-1).cpu().float(),
                return_pil_image=True,
            )

            track_conds = cond_tracks[:, :n_pokes]

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                # pad queries as the decoder expects 80, but we only have 40 GT tracks
                queries_eval = torch.cat([gt_queries, torch.rand_like(gt_queries) * 2 - 1], dim=0)
                gen_tracks = model.sample(
                    z,
                    points_per_traj=64,
                    query_pos=queries_eval.expand(K, -1, -1).to(device="cuda", dtype=torch.bfloat16),
                    track_conds=track_conds,
                    start_frame=start_frame.expand(K, -1, -1, -1).to(device="cuda", dtype=torch.bfloat16),
                )  # [B, N, T, 2]
                gen_tracks = gen_tracks[:, :n_gt_tracks]  # [B, 40, T, 2]

            our_viz = []
            for i in range(K):
                results[video_name][ours_name]["tracks"].append(gen_tracks[i].cpu().float())

                if noviz:
                    continue

                viz_tracks = (
                    gen_tracks[i].mul(0.5).add(0.5)
                    * torch.tensor([H - 1, W - 1], device=gen_tracks.device)[None, None, :]
                )
                viz = draw_trajectories_on_frame(
                    start_frame * 0.5 + 0.5,
                    viz_tracks.flip(-1).cpu().float(),
                    return_pil_image=True,
                )

                our_viz.append(viz)

            # save viz
            concatenate_images_horizontally([gt_viz, poke_viz] + our_viz).save(
                output_path / f"{video_name}-ours-cfg{cfg_scale}-n_pokes{n_pokes}-viz.png"
            )

        torch.save(results, output_path / "results.pt")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)

    # By launching with fire, all arguments become specifyable via the CLI
    fire.Fire(main)
