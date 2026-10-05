import math
from typing import Optional

import numpy as np
from torch.utils.data import Subset, DataLoader


def build_few_shot_subset(dataset, ratio: float, strategy: str = "uniform", seed: Optional[int] = None):
    ratio = max(1e-6, min(1.0, float(ratio)))
    if ratio >= 1.0 or len(dataset) <= 1:
        return dataset

    strategy = str(strategy).lower()
    if strategy not in {"begin", "end", "uniform", "random"}:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    total = len(dataset)
    keep = max(1, int(math.floor(total * ratio)))

    if strategy == "begin":
        indices = list(range(keep))
    elif strategy == "end":
        indices = list(range(total - keep, total))
    elif strategy == "uniform":
        internal = max(int(1.0 // ratio), 1)
        indices = [i * internal for i in range(keep) if (i * internal) < total]
        # if len(indices) < keep:
        #     extra = np.linspace(0, total - 1, num=keep, dtype=int).tolist()
        #     indices = sorted(set(indices + extra))[:keep]
    else:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(total, size=keep, replace=False)).tolist()

    return Subset(dataset, indices)

def apply_few_shot_loader(
    dataset,
    loader,
    config,
    *,
    drop_last: bool,
    collate_fn=None,
        
):
    mode = str(getattr(config, "shot_mode", "full_shot")).lower()
    if mode not in {"few_shot", "zero_shot", "full_shot"}:
        raise ValueError(f"Unknown shot_mode: {mode}")

    # zero/full: 不改 loader；few: 仅在 dataset_sample 模式做子集
    if mode != "few_shot":
        return loader

    subset = build_few_shot_subset(
        dataset,
        getattr(config, "sampling_rate", getattr(config, "few_shot_ratio", 0.1)),
        strategy=getattr(config, "sampling_strategy", "uniform"),
        seed=getattr(config, "seed", None),
    )

    return DataLoader(
        subset,
        batch_size=getattr(config, "batch_size", loader.batch_size),
        shuffle=True,
        num_workers=getattr(config, "num_workers", 0),
        drop_last=drop_last,
        collate_fn=collate_fn if collate_fn is not None else loader.collate_fn,
    )