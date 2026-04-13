# Learning Long-term Motion Embeddings for Efficient Kinematics Generation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://compvis.github.io/long-term-motion)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b)](https://arxiv.org/)
[![Weights](https://img.shields.io/badge/HuggingFace-Weights-orange)](https://huggingface.co/CompVis/ZipMo)
[![Venue](https://img.shields.io/badge/CVPR-2026-green)](#citation)

<div align="center">
  <a href="https://nickstracke.dev" target="_blank">Nick Stracke</a><sup>*,1,2</sup> · 
  <a href="https://koljabauer.com" target="_blank">Kolja Bauer</a><sup>*,1,2</sup> · 
  <a href="https://stefan-baumann.eu" target="_blank">Stefan Andreas Baumann</a><sup>1,2</sup> ·
  Miguel Angel Bautista<sup>3</sup> ·
  Josh Susskind<sup>3</sup> ·
  <a href="https://ommer-lab.com" target="_blank">Bjorn Ommer</a><sup>1,2</sup>
</div>
<p align="center">
  <b><sup>1</sup>CompVis @ LMU Munich, <sup>2</sup>MCML, <sup>3</sup>Apple</b><br/>
  CVPR 2026
</p>

<p align="center">
  <img src="docs/static/images/motion-embedding/teaser_white.png" alt="ZipMo teaser figure">
</p>

Understanding and predicting motion is a fundamental component of visual intelligence. ZipMo models scene dynamics by operating directly on a long-term motion embedding learned from large-scale tracker trajectories. This motion-first representation avoids full video synthesis when the target is kinematics: it supports efficient generation of long, realistic motions from spatial pokes in open-domain videos and from task/text embeddings in LIBERO.

## ✨ Highlights

- We learn a compact long-term motion embedding from tracker-derived trajectories and start-frame context.
- The embedding reaches 64x temporal compression and still supports dense reconstruction at arbitrary spatial query points.
- Conditional flow matching in this learned motion space generates diverse, goal-conditioned trajectories from spatial pokes and task/text embeddings.
- On open-domain videos and LIBERO robotics benchmarks, the model improves motion quality and action prediction while being substantially more efficient than video-space generation.

## 🚀 Usage

There are three common ways to get started:

1. Launch the interactive demo to try poke-conditioned motion generation visually.
2. Use Torch Hub if you want pretrained modules directly in your own code.
3. Install the repository manually if you want to run the demo, evaluation scripts, or LIBERO rollouts locally.

### 🎮 Interactive Demo

From a local checkout with dependencies installed, launch the Gradio demo. Need an environment first? Use the Manual Setup steps at the end of this section, then come back here.

```bash
# after Manual Setup below
python -m scripts.demo --server_port 55555
```

The demo loads the sparse planner, lets you upload or choose a start frame, click spatial pokes, and sample multiple plausible motion futures. Compilation is optional, but useful for repeated inference:

```bash
python -m scripts.demo --server_port 55555 --compile True
```

With compilation enabled, the first sampling step can be slow because PyTorch is compiling the model. Later samples should be faster.

### 🔥 Torch Hub

For programmatic use, the pretrained models are exposed through `hubconf.py`. Weights are downloaded automatically from [`CompVis/ZipMo`](https://huggingface.co/CompVis/ZipMo).

```python
import torch

repo = "CompVis/long-term-motion"

# Open-domain motion prediction
planner_sparse = torch.hub.load(repo, "zipmo_planner_sparse")
planner_dense = torch.hub.load(repo, "zipmo_planner_dense")

# Motion autoencoder
vae = torch.hub.load(repo, "zipmo_vae")

# LIBERO planning and policy components.
libero_atm_planner = torch.hub.load(repo, "zipmo_planner_libero", "atm")
libero_tramoe_planner = torch.hub.load(repo, "zipmo_planner_libero", "tramoe")
policy_head_atm = torch.hub.load(repo, "zipmo_policy_head", "atm")
policy_head_tramoe_goal = torch.hub.load(repo, "zipmo_policy_head", "tramoe", "goal")
```

Available pretrained entries:

- `zipmo_planner_sparse`: sparse-poke planner for open-domain video evaluation.
- `zipmo_planner_dense`: dense-conditioning planner for open-domain video evaluation.
- `zipmo_planner_libero`: LIBERO planner, with `mode` set to `atm` or `tramoe`.
- `zipmo_policy_head`: LIBERO policy head, with `mode` set to `atm` or `tramoe`; Tra-MoE also needs a suite name from `10`, `goal`, `object`, or `spatial`.
- `zipmo_vae`: the motion autoencoder used by the planners.

### 🛠️ Manual Setup

Clone the repository and install the Python dependencies:

```bash
git clone https://github.com/CompVis/long-term-motion.git
cd track-ae-release

conda create -n zipmo python=3.10 -y
conda activate zipmo
pip install -r requirements.txt
```

The default inference and evaluation paths assume a CUDA GPU and use `bfloat16`.

## 🔧 Training

**Data Preprocessing.** For data collection and preprocessing, follow the [Flow Poke Transformer data preprocessing guide](https://github.com/CompVis/flow-poke-transformer/blob/main/scripts/data/README.md). ZipMo uses the same general setup: collect videos, shard them with `webdataset`, and pre-extract tracker trajectories before training.

Single-GPU training can be launched with:

```bash
python -m scripts.train \
  --train_data_tar_base /path/to/preprocessed/train/shards \
  --val_data_tar_base /path/to/preprocessed/val/shards \
  --out_dir outputs/train_zipmo
```

For multi-GPU training, use `torchrun`:

```bash
torchrun --nnodes 1 --nproc-per-node 2 -m scripts.train \
  --train_data_tar_base /path/to/preprocessed/train/shards \
  --val_data_tar_base /path/to/preprocessed/val/shards \
  --out_dir outputs/train_zipmo
```

Training can be resumed from a checkpoint by adding, for example, `--load_checkpoint outputs/train_zipmo/checkpoints/checkpoint_0100000.pt`. All arguments in [`scripts/train.py`](scripts/train.py) are exposed through the CLI via `fire`.

## 📊 Track Prediction Evaluation

The standard open-domain evaluation has two stages:

1. Sample trajectories with [`scripts/sample.py`](scripts/sample.py).
2. Compute metrics with [`scripts/eval.py`](scripts/eval.py).

First download the evaluation targets and videos from the shared Google Drive folder:

https://drive.google.com/drive/folders/1ddt4gPIbfnvnARRjL1YdWYpGS0J3tASN

Place the files under `data/`. The sampler defaults to this layout:

```text
data/
  gt_tracks.pt
  pexels/
    original-<video_name>.mp4
    ...
```

If your downloaded folder uses a different layout, pass `--gt_path` and `--samples_path` explicitly.
For example, if the targets unpack into a `gt_data` folder, either place the target tensor at `data/gt_tracks.pt` or point `--gt_path` at the downloaded target file.

### 🧩 Sparse Mode

Sparse mode loads `zipmo_planner_sparse`, draws `K=8` samples per video, and evaluates the poke-conditioning settings `1`, `2`, `4`, `8`, and `16`. It also writes trajectory visualizations.

```bash
python -m scripts.sample --mode sparse

python -m scripts.eval \
  --results_path outputs/evals/sparse-cfg1.0-seed43/results.pt \
  --k 8
```

### 🧱 Dense Mode

Dense mode loads `zipmo_planner_dense`, conditions on all `40` target trajectories, and draws `K=128` samples. Visualization is disabled by default in this mode because the sample count is large.

```bash
python -m scripts.sample --mode dense

python -m scripts.eval \
  --results_path outputs/evals/dense-cfg1.0-seed43/results.pt \
  --k 128 # or 8
```

The evaluation prints per-model averages for `Min_MSE`, `Mean_MSE`, `MeanT_MSE`, endpoint error (`EPE`), and diversity statistics.

## 🤖 LIBERO Action Prediction Evaluation

For LIBERO, ZipMo predicts task-conditioned object motion from a task description and start frame. A lightweight policy head maps the generated motions to 7D robot actions for rollout evaluation.

The LIBERO simulation stack is more fragile than the open-domain video evaluation. The following setup was used for this release and has been tested on NVIDIA A100 GPUs.

```bash
conda create -n libero_env python=3.10 -y
conda activate libero_env

# From the project root:
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
cd LIBERO
pip install -r requirements.txt
pip install -e .
python benchmark_scripts/download_libero_datasets.py --download-dir .
cd ..

pip install -r requirements_libero.txt
python scripts/preproc_text_emb.py --dataset-root LIBERO/datasets
```

Then run policy evaluation:

```bash
cd <PROJECT_ROOT>
accelerate launch ./scripts/eval_libero_policy.py \
  --save_path <OUTPUT_DIR> \
  --ckpt_path <POLICY_CKPT_PATH> \
  --suite <libero_goal|libero_object|libero_spatial|libero_10|libero_90>
```

Required arguments are `--save_path`, `--ckpt_path`, and `--suite`. Defaults assume LIBERO is checked out at `<PROJECT_ROOT>/LIBERO`, with datasets at `LIBERO/datasets` and text embeddings at `LIBERO/task_embedding_caches/task_emb_bert.npy`. If your layout differs, also set `--dataset_path`, `--task_emb_cache_path`, and `--libero_path`.

`--ckpt_path` should point to a local policy-head checkpoint. The released policy heads live in the Hugging Face weights repository under `policy_heads/`, including `atm_libero.safetensors`, `tramoe_libero_10.safetensors`, `tramoe_libero_goal.safetensors`, `tramoe_libero_object.safetensors`, and `tramoe_libero_spatial.safetensors`.

Useful additional options:

```bash
accelerate launch ./scripts/eval_libero_policy.py \
  --save_path outputs/libero_eval \
  --ckpt_path <POLICY_CKPT_PATH> \
  --suite libero_goal \
  --num_env_rollouts 10 \
  --vec_env_num 10 \
  --track_pred_nfe 10 \
  --cfg_scale 1.0 \
  --vis_tracks
```

The script prints a final success-rate summary and writes rollout videos under `<OUTPUT_DIR>/eval_results/`.

## 🗂️ Repository Layout

```text
zipmo/
  vae.py          # long-term motion autoencoder
  planner.py      # sparse, dense, and LIBERO planners
  policy_head.py  # LIBERO action head
  blocks.py       # transformer and attention building blocks
  dino.py         # image encoder utilities

scripts/
  train.py                # motion autoencoder training
  demo.py                 # Gradio demo
  sample.py               # open-domain trajectory sampling
  eval.py                 # open-domain metric computation
  eval_libero_policy.py   # LIBERO rollout evaluation
  preproc_text_emb.py     # LIBERO task embedding preprocessing

docs/
  index.html       # project page
  static/          # figures, tables, and rollout videos

hubconf.py         # Torch Hub entry points
requirements.txt
requirements_libero.txt
```

## ⚙️ How It Works

ZipMo first learns a dense motion space by encoding sparse tracker trajectories and the start frame into a compact latent motion grid. The decoder can reconstruct dense trajectories at arbitrary query points.

![Dense motion-space learning](docs/static/images/motion-embedding/stage1_white.png)

It then trains a conditional flow-matching model in the learned motion space. At inference time, the planner samples a motion latent conditioned on scene context plus task information, such as spatial pokes or LIBERO task embeddings.

![ZipMo architecture](docs/static/images/motion-embedding/ZipMo_white.png)

## 🏆 Results

On open-domain videos, ZipMo is evaluated from sparse poke conditioning to dense guidance. The model is designed to generate kinematics directly, so it can compare against both specialized trajectory predictors and video generation baselines whose videos are tracked afterward.

![Comparison against motion predictors](docs/static/images/tab_1_screenshot.png)

For LIBERO action prediction, the model follows the ATM and Tra-MoE evaluation protocols and rolls out a policy head on top of generated motion.

![LIBERO action prediction results](docs/static/images/motion-embedding/action_pred_results.png)

## 🎓 Citation

If you find this code or model useful, please cite:

```bibtex
@inproceedings{stracke2026motionembeddings,
  title     = {Learning Long-term Motion Embeddings for Efficient Kinematics Generation},
  author    = {Stracke, Nick and Bauer, Kolja and Baumann, Stefan Andreas and Bautista, Miguel Angel and Susskind, Josh and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## 🙏 Acknowledgements

This repository uses project-page assets from the academic project page in `docs/`. The implementation builds on PyTorch, Torch Hub, Hugging Face Hub, Gradio, and the LIBERO robotics benchmark.
