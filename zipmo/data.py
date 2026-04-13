import glob
import io
import random
from functools import partial
from pathlib import Path
from typing import Literal

import decord
import numpy as np
import torch
import webdataset as wds
from jaxtyping import Bool, Float
from torchvision.transforms.functional import resize

decord.bridge.set_bridge("torch")


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True, **kwargs):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}  # remove keys with "__"

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = torch.tensor(np.array(list(batched[key])))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
            else:
                result[key] = list(batched[key])
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = torch.tensor(np.stack(list(batched[key])))
        else:
            result[key] = list(batched[key])

    return result


def get_shard_urls(shards: list[str] | tuple[str] | str):
    if isinstance(shards, (list, tuple)):
        patterns = shards
    else:  # a single string
        patterns = [shards]

    shard_urls = []
    for pat in patterns:
        p = Path(pat)
        if not p.parent.exists():
            print(f"Parent directory does not exist: {p.parent}", flush=True)
            continue

        if p.is_dir():
            matches = [str(tar) for tar in p.glob("*.tar")]
            print(f"Found {len(matches)} shards in directory: {pat}", flush=True)
        else:
            matches = glob.glob(str(p))
            print(f"Found {len(matches)} shards in pattern: {pat}", flush=True)
            assert len(matches) > 0, f"No shards matched pattern: {pat}"

        shard_urls.extend(matches)

    assert len(shard_urls) > 0, f"No shards found for {shards}"

    shard_urls = list(set(shard_urls))  # deduplicate
    shard_urls.sort()  # sort

    return shard_urls


def decode_npy(b: bytes):
    with io.BytesIO(b) as f:
        return torch.from_numpy(np.load(f, allow_pickle=True))


def get_video_decord(bytes: bytes, max_frames: int, start_idx: int = 0) -> Float[torch.Tensor, "T H W C"]:
    with io.BytesIO(bytes) as data_io:
        vr = decord.VideoReader(data_io)
        n_frames = min(len(vr) - start_idx, max_frames)
        video: Float[torch.Tensor, "T H W C"] = vr.get_batch(list(range(start_idx, start_idx + n_frames))) / 127.5 - 1.0
        return video


def get_frames_decord(video_bytes: bytes, indices: list[int]) -> Float[torch.Tensor, "N H W C"]:
    """Optimized decord for multiple frames"""
    if decord is None:
        raise ImportError("decord is required for get_frames_decord. Install it with `pip install decord`.")
    if not indices:
        raise ValueError("Indices list is empty")
    with io.BytesIO(video_bytes) as data_io:
        vr = decord.VideoReader(data_io)
        # get_batch handles arbitrary lists efficiently (duplicates, unsorted)
        frames = vr.get_batch(indices)  # [N, H, W, C]
        return frames / 127.5 - 1.0


class TrackerDataModule:
    def __init__(
        self,
        train: dict,
        validation: dict | None = None,
        n_encoder_trajectories: int = 40,
        n_decoder_trajectories: int | None = 40,
        n_samples: int = 64,  # number of samples per trajectory (frames)
        track_visibility: Literal[
            "full", "start"
        ] = "start",  # whether to use only fully visible tracks or also partially visible ones
        track_certainty: Literal[
            "full", "start"
        ] = "full",  # whether to use only tracks that are certain for all frames or just the start frame
        temporal_masking_ratio: float = 0.0,  # ratio of masked timesteps (for MAE training)
        frame_size: tuple[int, int] = (224, 224),  # size to resize the start frame to (if included)
        factor_top_magnitude_tracks: float = 0.0,  # factor for sorting trajectories by length and taking longest ones
        video_size: tuple[int, int] | None = (360, 640),  # size to resize the video frames to (H, W)
        filter_track_in_bounds: bool = False,
        static_camera_fraction_threshold: float = 0.5,  # if > 0, filter out samples where more than this fraction of tracks are static
    ):
        self.train = train
        self.validation = validation
        self.n_encoder_trajectories = n_encoder_trajectories
        self.n_decoder_trajectories = n_decoder_trajectories
        self.n_trajectories = n_encoder_trajectories + (0 if n_decoder_trajectories is None else n_decoder_trajectories)
        self.n_samples = n_samples
        self.track_visibility = track_visibility
        self.track_certainty = track_certainty
        self.frame_size = frame_size
        self.temporal_masking_ratio = temporal_masking_ratio
        self.factor_top_magnitude_tracks = factor_top_magnitude_tracks
        self.video_size = video_size
        self.filter_track_in_bounds = filter_track_in_bounds
        self.static_camera_fraction_threshold = static_camera_fraction_threshold

    def get_high_magnitude_tracks(
        self,
        tracks_yx: Float[torch.Tensor, "N T 2"],
    ):
        static_mask = self.get_static_tracks(tracks_yx)

        top_idxs = torch.nonzero(~static_mask).squeeze(dim=1)
        top_idxs = top_idxs[: int(self.factor_top_magnitude_tracks * self.n_trajectories)]

        all_indices = torch.arange(len(tracks_yx), device=tracks_yx.device)
        remaining_mask = ~torch.isin(all_indices, top_idxs)
        remaining_idxs = all_indices[remaining_mask]
        remaining_idxs = remaining_idxs[: self.n_trajectories - len(top_idxs)]

        idx = torch.cat([top_idxs, remaining_idxs])

        # Take the randomly selected trajectories
        tracks_yx = tracks_yx[idx]

        # shuffle before we return so top tracks are not always first
        # this is important for masking later on
        shuffled_tracks_yx = tracks_yx[torch.randperm(len(tracks_yx))]

        return shuffled_tracks_yx

    def decode(self, sample, decode_video=False):
        tracks_yx: Float[torch.Tensor, "T N 2"] = decode_npy(sample["tracks_yx.npy"])  # in [-1, 1]
        visibility = decode_npy(sample["visibility.npy"]).squeeze()

        T = tracks_yx.size(0)

        # if self.n_samples is None, we use all timesteps
        n_samples = self.n_samples if self.n_samples is not None else T
        if T < n_samples:
            return {"valid": False}

        start_idx = random.randint(0, max(0, T - n_samples))
        tracks_yx = tracks_yx[start_idx : start_idx + n_samples]
        visibility = visibility[start_idx : start_idx + n_samples]

        if "certainty.npy" in sample:
            certainty: Float[torch.Tensor, "T N"] = decode_npy(sample["certainty.npy"]).squeeze()
            certainty = certainty[start_idx : start_idx + n_samples]

            track_certain = certainty > 0.5  # (T, N)
            if self.track_certainty == "full":
                mask_certain = track_certain.all(dim=0)
            elif self.track_certainty == "start":
                mask_certain = track_certain[0] > 0.5
            else:
                raise ValueError(f"Unknown track_certainty: {self.track_certainty}")
        else:
            mask_certain = torch.ones(tracks_yx.shape[1], dtype=torch.bool)

        if self.filter_track_in_bounds:
            track_in_frame: Bool[torch.Tensor, "t n_t"] = (
                (tracks_yx[..., 0] >= -1)
                & (tracks_yx[..., 0] <= 1)
                & (tracks_yx[..., 1] >= -1)
                & (tracks_yx[..., 1] <= 1)
            )
        else:
            track_in_frame: Bool[torch.Tensor, "t n_t"] = torch.ones(
                tracks_yx.shape[0], tracks_yx.shape[1], dtype=torch.bool
            )

        track_visible = visibility > 0.5  # (T, N)
        if self.track_visibility == "full":
            mask_visible = track_visible.all(dim=0)
        elif self.track_visibility == "start":
            mask_visible = track_visible[0] > 0.5
        else:
            raise ValueError(f"Unknown track_visibility: {self.track_visibility}")

        mask = mask_certain & mask_visible & track_in_frame.all(dim=0)  # (N,)

        # switch dims so now time is second dim
        tracks_yx: Float[torch.Tensor, "N T 2"] = tracks_yx.movedim(0, 1)[mask]

        # make sure enough left over after masking
        if tracks_yx.shape[0] < self.n_trajectories:
            return {"valid": False}

        # shuffle tracks before further selection
        tracks_yx = tracks_yx[torch.randperm(len(tracks_yx))]

        tracks_yx = self.get_high_magnitude_tracks(tracks_yx)[: self.n_trajectories]

        tracks_enc_yx = tracks_yx[: self.n_encoder_trajectories]
        tracks_dec_yx = tracks_yx

        out_dict = {
            "tracks_enc_yx": tracks_enc_yx,
            "tracks_dec_yx": tracks_dec_yx,  # encoder + decoder tracks
        }

        if decode_video:
            video = get_video_decord(
                sample["video.mp4"], max_frames=n_samples, start_idx=start_idx
            )  # (T, H, W, C) in [-1, 1]

            if self.video_size is not None:
                video = resize(video.movedim(-1, 0), list(self.video_size)).movedim(0, -1)
            out_dict["video"] = video
            out_dict["start_frame"] = video[0]
        else:
            out_dict["start_frame"] = get_frames_decord(sample["video.mp4"], [start_idx])[0]  # (H, W, C) in [-1, 1]

        out_dict["start_frame"] = resize(out_dict["start_frame"].movedim(-1, 0), list(self.frame_size)).movedim(0, -1)

        return out_dict

    def filter(self, sample):
        return sample.get("valid", True)

    def get_static_tracks(self, tracks_yx: Float[torch.Tensor, "T N 2"]) -> Bool[torch.Tensor, "N"]:
        # divide by two to normalize because tracks are in [-1,1]
        flow = torch.diff(tracks_yx, dim=0) / 2.0
        flow_magnitude = flow.norm(dim=-1)

        return (flow_magnitude < 0.01).float().mean(dim=0) > 0.95

    def get_camera_static(
        self,
        sample: dict,
    ) -> bool:
        static_tracks = self.get_static_tracks(decode_npy(sample["tracks_yx.npy"]))
        is_static = static_tracks.float().mean().item() > self.static_camera_fraction_threshold

        return is_static

    def get_loader(
        self,
        shards: list[str],
        batch_size: int,
        num_workers: int,
        prefetch_factor: int = 2,
        shuffle=0,
        repeat_shards=False,
        repeat_shards_deterministic=False,
        shard_detshuffle=False,
        decode_video=False,
    ):
        shard_urls = get_shard_urls(shards)

        dataset = wds.DataPipeline(
            (
                wds.SimpleShardList(urls=shard_urls)
                if not repeat_shards
                else wds.ResampledShards(urls=shard_urls, deterministic=repeat_shards_deterministic)
            ),
            wds.detshuffle() if shard_detshuffle else wds.shuffle(),
            wds.split_by_node,
            wds.split_by_worker,
            partial(wds.tarfile_samples, handler=wds.warn_and_continue),
            *([wds.shuffle(shuffle)] if shuffle != 0 else []),
            wds.select(self.get_camera_static),
            wds.map(partial(self.decode, decode_video=decode_video)),
            wds.select(self.filter),
            wds.batched(batch_size, partial=False, collation_fn=dict_collation_fn),
        )

        return wds.WebLoader(
            dataset,
            batch_size=None,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self.get_loader(
            **self.train,
            shuffle=100,
            repeat_shards=True,
            shard_detshuffle=True,
        )

    def val_dataloader(self):
        assert self.validation is not None, "Validation config is not provided"
        return self.get_loader(
            **self.validation,
            repeat_shards=True,
            repeat_shards_deterministic=True,
            shard_detshuffle=True,
            decode_video=True,  # nice for visualization
        )

    def test_dataloader(self):
        return self.val_dataloader()
