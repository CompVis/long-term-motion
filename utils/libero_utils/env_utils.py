# Adapted from https://github.com/pairlab/AMPLIFY
import logging
import multiprocessing
import os
import random
import re
import sys
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch

# Required for vectorized envs
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


def seed_everything(seed: int) -> None:
    """Seed Python / NumPy / Torch RNGs for reproducible eval."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def merge_results(results: list[dict]):
    merged = {}
    for result in results:
        merged.update(result)

    def _collect_env_metric(prefix: str) -> np.ndarray:
        # Only include per-environment keys like rollout/<metric>_env0, _env1, ...
        # Exclude aggregate keys such as rollout/<metric>_env_avg.
        pattern = re.compile(rf"^{re.escape(prefix)}\d+$")
        vals = [v for k, v in merged.items() if pattern.match(k)]
        return np.array(vals, dtype=np.float32)

    returns = _collect_env_metric("rollout/return_env")
    horizons = _collect_env_metric("rollout/horizon_env")
    successes = _collect_env_metric("rollout/success_env")

    merged["rollout/return_env_avg"] = float(np.mean(returns))
    merged["rollout/horizon_env_avg"] = float(np.mean(horizons))
    merged["rollout/success_env_avg"] = float(np.mean(successes))
    return merged


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_libero_repo_on_path() -> None:
    """Make `import libero` work when LIBERO is cloned into this repo."""
    local_libero_root = _repo_root() / "LIBERO"
    if local_libero_root.exists():
        local_libero_root_str = str(local_libero_root)
        if local_libero_root_str not in sys.path:
            sys.path.insert(0, local_libero_root_str)


def _get_libero_imports():
    _ensure_libero_repo_on_path()
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `libero`. Run the setup in Libero_README.md "
            "and ensure `LIBERO/` exists at the project root."
        ) from exc
    return benchmark, get_libero_path, OffScreenRenderEnv


def _task_name_to_language(task_name: str) -> str:
    """Match ATM task-name parsing used for BERT task embedding caches."""
    name = task_name.replace("_demo", "")
    if name and name[0].isupper():  # LIBERO-10 and LIBERO-90 style
        if "SCENE10" in name:
            return " ".join(name[name.find("SCENE") + 8 :].split("_"))
        return " ".join(name[name.find("SCENE") + 7 :].split("_"))
    return " ".join(name.split("_"))


def _resolve_init_states_path(task, libero_path: str | None, get_libero_path) -> str:
    """
    Resolve init-state path from the explicit local LIBERO clone first.
    """
    candidates: list[Path] = []
    if libero_path is not None:
        candidates.append(Path(libero_path).expanduser() / "init_files" / task.problem_folder / task.init_states_file)

    candidates.append(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file)

    for path in candidates:
        if path.exists():
            return str(path)

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find LIBERO init-state file `{task.init_states_file}` for suite `{task.problem_folder}`. "
        f"Searched: {searched}"
    )


def _load_task_init_states(benchmark_instance, task_no: int, get_libero_path, libero_path: str | None = None):
    """
    Load LIBERO task init states in a PyTorch-version compatible way.
    """
    task = benchmark_instance.get_task(task_no)
    init_states_path = _resolve_init_states_path(
        task=task,
        libero_path=libero_path,
        get_libero_path=get_libero_path,
    )
    # Be explicit for PyTorch >=2.6 where the default changed to weights_only=True.
    # These LIBERO init-state files are trusted local data and require full unpickling.
    try:
        return torch.load(init_states_path, weights_only=False)
    except TypeError:
        # Older PyTorch versions may not support the weights_only argument.
        return torch.load(init_states_path)


@lru_cache(maxsize=8)
def _load_task_emb_cache(cache_path: str) -> dict[str, np.ndarray]:
    cache = np.load(cache_path, allow_pickle=True).item()
    if not isinstance(cache, dict):
        raise ValueError(f"Task embedding cache must be a dict, got {type(cache)} from {cache_path}")
    return cache


def get_task_emb(
    task_suite,
    task_name,
    dataset_path=None,
    task_language=None,
    task_emb_cache_path=None,
):
    """
    Returns the task embedding for a given task.

    Returns:
        task_emb: torch.Tensor, task embedding
    """
    demo_root: str
    if dataset_path is None:
        _, get_libero_path, _ = _get_libero_imports()
        demo_root = get_libero_path("datasets")

        if not os.path.exists(demo_root):
            raise ValueError(
                f"LIBERO dataset not found at {demo_root}, please set dataset_path in your"
                " respective config or ensure datasets are in the standard location:"
                " /LIBERO/libero/datasets"
            )
    else:
        demo_root = os.path.expanduser(dataset_path)
        if not os.path.exists(demo_root):
            raise ValueError(f"Provided dataset_path does not exist: {demo_root}")

    task_suite_path = os.path.join(demo_root, task_suite)
    task_file = f"{task_name}_demo.hdf5"
    task_file_path = os.path.join(task_suite_path, task_file)
    if not os.path.exists(task_file_path):
        raise FileNotFoundError(
            f"Task demo file not found: {task_file_path}. "
            f"Check task_suite={task_suite}, task_name={task_name}, dataset_path={demo_root}."
        )
    cache_path = (
        Path(task_emb_cache_path)
        if task_emb_cache_path
        else _repo_root() / "LIBERO/task_embedding_caches/task_emb_bert.npy"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Required task embedding cache not found at {cache_path}. "
            "Run `python scripts/preproc_text_emb.py --dataset-root LIBERO/datasets` before building envs."
        )

    language_key = task_language if task_language is not None else _task_name_to_language(task_name)
    cache = _load_task_emb_cache(str(cache_path))
    if language_key not in cache:
        raise KeyError(
            f"Task language `{language_key}` not found in cache {cache_path}. "
            "Rebuild cache with current datasets: "
            "`python scripts/preproc_text_emb.py --dataset-root LIBERO/datasets --overwrite`."
        )
    task_emb = torch.tensor(cache[language_key], dtype=torch.float32)

    return task_emb


def build_libero_env(
    task_suite,
    task_no,
    img_size,
    dataset_path,
    action_dim=7,
    n_envs=1,
    use_depth=False,
    segmentation_level=None,
    flip_image=True,
    libero_path=None,
    task_emb_cache_path=None,
    seed: int | None = None,
    vecenv=True,
    **kwargs,
):
    """
    Builds a libero environment.

    Returns:
        env: (vectorized) environment
        task: str, task description
        task_emb: torch.Tensor, task embedding
    """
    assert action_dim in [4, 7], "Only 4 or 7 action dimensions are supported"
    logging.info("Building LIBERO environment...")
    benchmark, get_libero_path, OffScreenRenderEnv = _get_libero_imports()
    from .wrappers import (
        EnvStateWrapper,
        FourDOFWrapper,
        LiberoImageUpsideDownWrapper,
        LiberoObservationWrapper,
        LiberoResetWrapper,
        LiberoSuccessWrapper,
        StackDummyVectorEnv,
        StackSubprocVectorEnv,
    )

    if libero_path is None:
        libero_path = get_libero_path("benchmark_root")

    # initialize a benchmark
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[task_suite]()
    task = benchmark_instance.get_task(task_no)

    # Task embedding
    task_emb = get_task_emb(
        task_suite=task_suite,
        task_name=task.name,
        dataset_path=dataset_path,
        task_language=task.language,
        task_emb_cache_path=task_emb_cache_path,
    )
    init_states = _load_task_init_states(
        benchmark_instance=benchmark_instance,
        task_no=task_no,
        get_libero_path=get_libero_path,
        libero_path=libero_path,
    )

    env_args = {
        "bddl_file_name": os.path.join(libero_path, "bddl_files", task.problem_folder, task.bddl_file),
        "camera_heights": img_size,
        "camera_widths": img_size,
        "ignore_done": True,
        "camera_depths": use_depth,
        "camera_segmentations": segmentation_level,  # None, 'instance', 'class', 'element'
        "camera_names": [
            "agentview",
            "robot0_eye_in_hand",
        ],  # ['frontview', 'birdview', 'agentview', 'sideview', 'galleryview', 'robot0_robotview', 'robot0_eye_in_hand'],
    }
    env_args.update(kwargs)

    def env_func(init_state_no=0, worker_seed: int | None = None):
        if worker_seed is not None:
            # Keep Python / NumPy / Torch RNG in each worker deterministic.
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        env = OffScreenRenderEnv(**env_args)
        env = LiberoResetWrapper(env, init_states=init_states, init_state_no=init_state_no)
        env = EnvStateWrapper(env)
        if action_dim == 4:
            env = FourDOFWrapper(env)
        if flip_image:
            env = LiberoImageUpsideDownWrapper(env)
        env = LiberoSuccessWrapper(env)
        env = LiberoObservationWrapper(env, masks=None, cameras=env_args["camera_names"])
        env.seed(worker_seed if worker_seed is not None else init_state_no)
        return env

    if vecenv:
        rng = random.Random(seed) if seed is not None else random
        init_state_count = len(init_states)
        if n_envs <= init_state_count:
            init_state_no = rng.sample(range(init_state_count), n_envs)
        else:
            init_state_no = [rng.randrange(init_state_count) for _ in range(n_envs)]

        worker_seeds = (
            [seed + i for i in range(n_envs)]
            if seed is not None
            else [init_state_no[i] for i in range(n_envs)]
        )
        if n_envs == 1:
            env = StackDummyVectorEnv([partial(env_func, init_state_no[0], worker_seeds[0])])
        else:
            env = StackSubprocVectorEnv([partial(env_func, init_state_no[i], worker_seeds[i]) for i in range(n_envs)])
    else:
        assert n_envs == 1, "Non-vectorized environment can only have one environment"
        init_state_no = 0 if len(init_states) == 0 else (seed % len(init_states) if seed is not None else 0)
        env = env_func(init_state_no=init_state_no, worker_seed=seed)

    return env, task.language, task_emb
