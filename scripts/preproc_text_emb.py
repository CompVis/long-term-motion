#!/usr/bin/env python3
"""
Minimal LIBERO task-text embedding preprocessor.

This reproduces the BERT task embedding cache behavior from ATM's
`scripts/preprocess_libero.py` without any CoTracker / trajectory processing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def get_task_name_from_file_name(file_name: str) -> str:
    """
    Match ATM's task-name parsing logic.

    Args:
        file_name: file stem without extension (e.g., "..._demo" suffix included)
    """
    name = file_name.replace("_demo", "")
    if name and name[0].isupper():  # LIBERO-10 and LIBERO-90 naming style
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def collect_task_names(dataset_root: Path) -> list[str]:
    h5_files = sorted(dataset_root.glob("*/*_demo.hdf5"))
    if not h5_files:
        raise FileNotFoundError(
            f"No *_demo.hdf5 files found under {dataset_root}. "
            "Expected LIBERO dataset structure like <root>/<suite>/*.hdf5."
        )
    task_names = sorted({get_task_name_from_file_name(path.stem) for path in h5_files})
    return task_names


def compute_task_embs(
    task_names: list[str],
    model_name: str = "bert-base-cased",
    max_word_len: int = 25,
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()

    tokens = tokenizer(
        text=task_names,
        add_special_tokens=True,
        max_length=max_word_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].cpu().numpy()

    return {task_names[i]: task_embs[i] for i in range(len(task_names))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute LIBERO BERT task embeddings cache.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("LIBERO/datasets"),
        help="Root containing LIBERO suites (e.g., libero_spatial, libero_object, ...).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("LIBERO/task_embedding_caches/task_emb_bert.npy"),
        help="Output .npy path (dict: task language string -> embedding).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-cased",
        help="HF model name used for text embeddings.",
    )
    parser.add_argument(
        "--max-word-len",
        type=int,
        default=25,
        help="Tokenizer max length; matches ATM preprocessing default.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./data/bert",
        help="HuggingFace cache directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )

    task_names = collect_task_names(dataset_root)
    task_name_to_emb = compute_task_embs(
        task_names=task_names,
        model_name=args.model_name,
        max_word_len=args.max_word_len,
        cache_dir=args.cache_dir,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, task_name_to_emb)

    emb_dim = len(next(iter(task_name_to_emb.values())))
    print(f"Saved {len(task_name_to_emb)} task embeddings (dim={emb_dim}) to: {output_path}")
    print(f"Sample task: {task_names[0]}")


if __name__ == "__main__":
    main()
