#!/usr/bin/env python3
"""Materialize a local WPBench dataset into dataset/forecasting/."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _copytree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "wandb"))


def _hardlink_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy/symlink/hardlink a local WPBench dataset into dataset/forecasting/.")
    parser.add_argument("--source", required=True, help="Source dataset directory, e.g. <original_checkout>/dataset/forecasting/forecasting_wp1_all_shapes_v1")
    parser.add_argument("--dataset-id", default="forecasting_wp1_all_shapes_v1")
    parser.add_argument("--mode", choices=["copy", "symlink", "hardlink"], default="copy")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--artifact-root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(src)
    artifact_root = Path(args.artifact_root).resolve()
    dst = artifact_root / "dataset" / "forecasting" / args.dataset_id

    if args.mode == "copy":
        _copytree(src, dst, args.overwrite)
    elif args.mode == "hardlink":
        _hardlink_tree(src, dst, args.overwrite)
    else:
        if dst.exists() or dst.is_symlink():
            if not args.overwrite:
                raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
            if dst.is_dir() and not dst.is_symlink():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src, target_is_directory=True)

    print(f"materialized {src} -> {dst} mode={args.mode}")


if __name__ == "__main__":
    main()
