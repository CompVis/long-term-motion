from collections import defaultdict

import fire
import numpy as np
import torch
from jaxtyping import Float


def inpaint_nans_tracks(their_tracks: list[torch.Tensor], gt_tracks: torch.Tensor) -> list[torch.Tensor]:
    """
    Forward-fill NaNs in [N, T, 2] tracks for each sample.
    If the first timestep is NaN, use GT at t=0.
    A timestep is considered valid only if both coords are finite.
    """
    if not their_tracks:
        return their_tracks

    tracks = torch.stack(their_tracks)  # [K, N, T, 2]
    K, N, T, _ = tracks.shape
    gt = gt_tracks.expand(K, -1, -1, -1)  # [K, N, T, 2]

    # t=0: if NaN, copy from GT
    first = tracks[:, :, 0, :]
    first_isnan = torch.isnan(first)
    tracks[:, :, 0, :] = torch.where(first_isnan, gt[:, :, 0, :], first)

    # valid mask: both coords finite
    valid = torch.isfinite(tracks[..., 0]) & torch.isfinite(tracks[..., 1])  # [K, N, T]

    # running last-valid index via cummax over indices

    idx = torch.arange(T, device=tracks.device).view(1, 1, T).expand(K, N, T)
    last_idx = torch.where(valid, idx, torch.full_like(idx, -1))
    last_idx, _ = torch.cummax(last_idx, dim=2)  # [K, N, T]; stays at last seen valid (or -1)

    # We guaranteed t=0 is valid after the GT fix, so -1 shouldn't occur. Clamp for safety.
    assert (last_idx >= 0).all(), "Some tracks have no valid points even after GT fill!"
    gather_idx = last_idx.clamp_min(0).to(torch.long)  # [K, N, T]

    # Gather along time dimension for both coords
    gather_idx_exp = gather_idx.unsqueeze(-1).expand(K, N, T, 2)  # [K, N, T, 2]
    filled = torch.gather(tracks, dim=2, index=gather_idx_exp)  # [K, N, T, 2]

    return list(torch.unbind(filled, dim=0))


def compute_metrics_atm(
    gen_tracks: list[Float[torch.Tensor, "K N T 2"]],
    gt_tracks: list[Float[torch.Tensor, "N T 2"]],
    mse_res: tuple[int, int] = (128, 128),
    normalize_by_track_magnitude: bool = False,
    return_per_batch: bool = False,
    include_l1: bool = False,
) -> dict[str, float] | dict[str, list[float | int]]:
    H, W = mse_res

    mean_T_mse_list = []
    mean_mse_list = []
    min_mse_list = []
    min_mse_idx_list = []

    mean_T_l1_list = []
    mean_l1_list = []
    min_l1_list = []
    min_l1_idx_list = []
    l1_dict = {}

    for gt_traj, gen_traj in zip(gt_tracks, gen_tracks):
        gt_traj = ((gt_traj + 1) / 2) * torch.tensor([H, W], device=gt_traj.device)  # map to pixel coords
        gen_traj = ((gen_traj + 1) / 2) * torch.tensor([H, W], device=gt_traj.device)  # map to pixel coords

        # compute track magnitude of the gt tracks
        track_magnitude = torch.ones((gt_traj.shape[0],), device=gt_traj.device, dtype=gt_traj.dtype)  # [N,]
        if normalize_by_track_magnitude:
            gt_magnitude = torch.diff(gt_traj, dim=1).pow(2).sum(-1).sqrt()  # [N, T-1]
            track_magnitude = gt_magnitude.sum(dim=-1) + 1  # [N,]

        mean_traj = gen_traj.mean(dim=0)  # [N, T, 2]

        meanT_mse = ((mean_traj - gt_traj) / track_magnitude[:, None, None]).pow(2).mean()  # []

        mse = ((gen_traj - gt_traj[None, ...]) / track_magnitude[None, :, None, None]).pow(2).mean([1, 2, 3])  # [k]

        # print(f"{meanT_mse.shape=} ; {mse.shape=}", flush=True)

        min_mse, min_idx = mse.min(dim=0)
        mean_mse = mse.mean(dim=0)

        mean_T_mse_list.append(meanT_mse.cpu().item())
        mean_mse_list.append(mean_mse.cpu().item())
        min_mse_list.append(min_mse.cpu().item())
        min_mse_idx_list.append(min_idx.cpu().item())

        if include_l1:
            meanT_l1 = ((mean_traj - gt_traj) / track_magnitude[:, None, None]).abs().mean()  # []

            l1 = ((gen_traj - gt_traj[None, ...]) / track_magnitude[None, :, None, None]).abs().mean([1, 2, 3])  # [k]

            min_l1, min_idx_l1 = l1.min(dim=0)
            mean_l1 = l1.mean(dim=0)

            mean_T_l1_list.append(meanT_l1.cpu().item())
            mean_l1_list.append(mean_l1.cpu().item())
            min_l1_list.append(min_l1.cpu().item())
            min_l1_idx_list.append(min_idx_l1.cpu().item())
            l1_dict = {
                "MeanT_L1": mean_T_l1_list,
                "Mean_L1": mean_l1_list,
                "Min_L1": min_l1_list,
                "Min_Idx_L1": min_l1_idx_list,
            }

    if return_per_batch:
        return {
            "MeanT_MSE": mean_T_mse_list,
            "Mean_MSE": mean_mse_list,
            "Min_MSE": min_mse_list,
            "Min_Idx": min_mse_idx_list,
            **l1_dict,
        }

    meanT_mse_overall = np.array(mean_T_mse_list).mean()
    mean_mse_overall = np.array(mean_mse_list).mean()
    min_mse_overall = np.array(min_mse_list).mean()

    return {"MeanT_MSE": meanT_mse_overall, "Mean_MSE": mean_mse_overall, "Min_MSE": min_mse_overall}


def compute_metrics(results, inpaint_nans=False, k=8):
    metrics_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    video_order = []
    for video in results:
        video_order.append(video)

        for model in results[video]:
            if model == "gt" or model == "original":
                continue

            gt_tracks = results[video]["gt"]["tracks"]  # [N, T, 2]
            their_tracks = results[video][model]["tracks"][:k]  # list[ [N, T, 2] ]
            # metrics_dict["video_order"].append(video)

            if isinstance(gt_tracks, np.ndarray):
                gt_tracks = torch.from_numpy(gt_tracks)
            their_tracks = [torch.from_numpy(t) if isinstance(t, np.ndarray) else t for t in their_tracks]

            gt_n = gt_tracks.shape[0]
            gt_t = gt_tracks.shape[1]

            if "pokes" in model:
                n_pokes = int(model.split("-")[-1].split("_")[0])
                print(f"using {n_pokes} pokes for eval")
            else:
                print("using all pokes for eval")
                n_pokes = gt_n

            if inpaint_nans:
                their_tracks = inpaint_nans_tracks(their_tracks, gt_tracks)
            else:
                # check if any nans in tracks
                for i in range(len(their_tracks)):
                    if torch.isnan(their_tracks[i]).any():
                        print(f"Warning: {model} has NaNs in track {i} for video {video}!")
                        assert False, "NaNs found in tracks!"

            their_t = their_tracks[0].shape[-2]

            print("their_t:", their_t, "gt_t:", gt_t)

            if their_t > gt_t:
                print(f"{model}: interpolating down their tracks")
                their_tracks = [
                    (torch.nn.functional.interpolate(t.permute(0, 2, 1), size=gt_t, mode="linear").permute(0, 2, 1))
                    for t in their_tracks
                ]
            elif their_t < gt_t:
                print(f"{model}: interpolating down gt tracks tp {their_t}")
                gt_tracks = torch.nn.functional.interpolate(
                    gt_tracks.permute(0, 2, 1), size=their_t, mode="linear"
                ).permute(0, 2, 1)
            else:
                print(f"{model}: no interpolation needed")

            their_tracks = torch.stack(their_tracks)  # [K, N, T, 2]

            print(model, "any nans:", torch.isnan(their_tracks).any())

            res = compute_metrics_atm([their_tracks], [gt_tracks], return_per_batch=True)

            # Important!! endpoint diff only over the poked points!!
            theirs_scaled = their_tracks.mul(0.5).add(0.5).mul(128)
            gt_scaled = gt_tracks.mul(0.5).add(0.5).mul(128)
            endpoint_diff = (
                torch.norm((theirs_scaled[:, :n_pokes, -1] - gt_scaled[:n_pokes, -1][None]), dim=-1).mean().item()
            )
            res["EPE"] = [endpoint_diff]
            res["video_name"] = [video]

            std_pos = torch.std(theirs_scaled, dim=0)  # [N, T, 2] - std across K samples
            diversity = torch.mean(std_pos).item()  # Scalar average std
            res["std_diversity"] = [diversity]

            # Flatten each trajectory to [K, N*T*2], compute pairwise distances
            K = theirs_scaled.shape[0]
            # flat_tracks = theirs_scaled[:, :n_pokes].view(K, -1)  # [K, N*T*2]
            flat_tracks = theirs_scaled.view(K, -1)  # [K, N*T*2]
            pairwise_dists = torch.cdist(flat_tracks, flat_tracks)  # [K, K]
            diversity_pairwise = torch.mean(pairwise_dists).item()  # Average distance
            res["diversity_pairwise"] = [diversity_pairwise]

            metrics_dict[model]["metrics"]["results"].append(res)
            res_normalized = compute_metrics_atm(
                [their_tracks], [gt_tracks], normalize_by_track_magnitude=True, return_per_batch=True
            )
            res_normalized["video_name"] = [video]
            metrics_dict[model]["metrics"]["results_normalized"].append(
                res_normalized
                # compute_metrics_libero_atm_nan([their_tracks], [gt_tracks], normalize_by_track_magnitude=True, strict_frame_wise=True)
            )

    return metrics_dict, video_order


def print_metrics(metrics_dict):
    # loop over models
    for model, results in metrics_dict.items():
        print(f"=== Model: {model} ===")
        # loop over metric types
        for name, metrics in results["metrics"].items():
            print(name)
            # print(sorted([(m['Min_MSE'],m['video_name']) for m in metrics], key=lambda x: x[0], reverse=True))
            avg_metrics = {
                "Min_MSE": sum(m["Min_MSE"][0] for m in metrics) / len(metrics),
                "Mean_MSE": sum(m["Mean_MSE"][0] for m in metrics) / len(metrics),
                "MeanT_MSE": sum(m["MeanT_MSE"][0] for m in metrics) / len(metrics),
                "EPE": sum(m["EPE"][0] for m in metrics if "EPE" in m) / len(metrics),
                "std_diversity": sum(m["std_diversity"][0] for m in metrics if "std_diversity" in m) / len(metrics),
                "diversity_pairwise": sum(m["diversity_pairwise"][0] for m in metrics if "diversity_pairwise" in m)
                / len(metrics),
            }
            if "physics_iq" in metrics[0]:
                avg_metrics["physics_iq_spatial_iou"] = sum(
                    m["physics_iq"]["spatial_iou"].item() for m in metrics
                ) / len(metrics)
                avg_metrics["physics_iq_spatiotemporal_iou"] = sum(
                    m["physics_iq"]["spatiotemporal_iou"].item() for m in metrics
                ) / len(metrics)
                avg_metrics["physics_iq_weighted_spatial_iou"] = sum(
                    m["physics_iq"]["weighted_spatial_iou"].item() for m in metrics
                ) / len(metrics)
                avg_metrics["physics_iq_mse"] = sum(m["physics_iq"]["mse"].item() for m in metrics) / len(metrics)
            print(f"  Metrics Type: {name}: {avg_metrics}")


def main(
    results_path: str = "./outputs/evals/dense-cfg1.0-seed43/results.pt",
    k: int = 8,
    inpaint_nans: bool = False,
):
    results_dict = torch.load(results_path, weights_only=False, map_location="cpu")
    metrics_dict, video_names = compute_metrics(results_dict, inpaint_nans=inpaint_nans, k=k)
    print_metrics(metrics_dict)


if __name__ == "__main__":
    fire.Fire(main)
