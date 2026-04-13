# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025 Nick Stracke et al., CompVis @ LMU Munich
from typing import Literal

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.hub import get_dir

dependencies = [
    "einops",
    "huggingface_hub",
    "jaxtyping",
    "safetensors",
    "torch",
    "tqdm",
]

_HF_REPO_ID = "CompVis/ZipMo"

_MODEL_FILES = {
    "zipmo_planner_dense": "zipmo_planner_dense.safetensors",
    "zipmo_planner_sparse": "zipmo_planner_sparse.safetensors",
    "zipmo_planner_libero_atm": "zipmo_planner_libero_atm.safetensors",
    "zipmo_planner_libero_tramoe": "zipmo_planner_libero_tramoe.safetensors",
    "zipmo_vae": "zipmo_vae.safetensors",
    "zipmo_policy_head_atm": "policy_heads/atm_libero.safetensors",
    "zipmo_policy_head_tramoe_10": "policy_heads/tramoe_libero_10.safetensors",
    "zipmo_policy_head_tramoe_goal": "policy_heads/tramoe_libero_goal.safetensors",
    "zipmo_policy_head_tramoe_object": "policy_heads/tramoe_libero_object.safetensors",
    "zipmo_policy_head_tramoe_spatial": "policy_heads/tramoe_libero_spatial.safetensors",
}


def _download_safetensors(filename: str) -> str:
    hub_dir = get_dir()

    if "/" in filename:
        subfolder, local_name = filename.rsplit("/", 1)
    else:
        subfolder, local_name = None, filename

    return hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=local_name,
        subfolder=subfolder,
        cache_dir=hub_dir,
    )


def zipmo_planner_dense(pretrained: bool = True, **kwargs):
    from zipmo.planner import ZipMoPlanner_Dense
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    model = ZipMoPlanner_Dense(vae=vae, **kwargs)

    if pretrained:
        path = _download_safetensors(_MODEL_FILES["zipmo_planner_dense"])
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_planner_sparse(pretrained: bool = True, **kwargs):
    from zipmo.planner import ZipMoPlanner_Sparse
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    model = ZipMoPlanner_Sparse(vae=vae, **kwargs)

    if pretrained:
        path = _download_safetensors(_MODEL_FILES["zipmo_planner_sparse"])
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_planner_libero(mode: Literal["atm", "tramoe"], pretrained: bool = True, **kwargs):
    from zipmo.planner import ZipMoPlanner_Libero_ATM, ZipMoPlanner_Libero_TraMoE
    from zipmo.vae import ZipMoVAE

    vae = ZipMoVAE()
    assert mode in ["atm", "tramoe"], "Mode must be either 'atm' or 'tramoe'"
    planner_cls = ZipMoPlanner_Libero_ATM if mode == "atm" else ZipMoPlanner_Libero_TraMoE
    model = planner_cls(vae=vae, **kwargs)

    name = f"zipmo_planner_libero_{mode}"

    if pretrained:
        path = _download_safetensors(_MODEL_FILES[name])
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_policy_head(
    mode: Literal["atm", "tramoe"],
    suite: Literal["10", "goal", "object", "spatial"] | None = None,
    pretrained: bool = True,
    **kwargs,
):
    assert mode == "atm" or suite is not None, "For TraMOE, a suite must be specified"
    assert mode in ["atm", "tramoe"], "Mode must be either 'atm' or 'tramoe'"

    from zipmo.policy_head import PolicyHeadATM, PolicyHeadTraMoE

    policy_cls = PolicyHeadATM if mode == "atm" else PolicyHeadTraMoE

    model = policy_cls(**kwargs)

    name = f"zipmo_policy_head_{mode}"
    if mode == "tramoe":
        name += f"_{suite}"

    if pretrained:
        path = _download_safetensors(_MODEL_FILES[name])
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model


def zipmo_vae(pretrained: bool = True, **kwargs):
    from zipmo.vae import ZipMoVAE

    model = ZipMoVAE(**kwargs)

    if pretrained:
        path = _download_safetensors(_MODEL_FILES["zipmo_vae"])
        state_dict = load_file(path)
        model.load_state_dict(state_dict)

    return model
