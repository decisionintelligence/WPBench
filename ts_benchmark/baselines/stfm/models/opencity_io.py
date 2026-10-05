"""OpenCity tensor and graph bridges for the benchmark."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.stfm.submodules.opencity.predifineGraph import cal_lape

OPENCITY_CONTEXT = 288
OPENCITY_HORIZON = 288
OPENCITY_DATASET_KEY = "METR_LA"
OPENCITY_CHECKPOINT = os.environ.get(
    "WPBENCH_OPENCITY_CHECKPOINT",
    os.path.join(
        os.path.abspath(os.path.join(__file__, "..", "..", "..")),
        "checkpoints",
        "OpenCity-plus.pth",
    ),
)


def _as_numpy_adjacency(adj_mx: Any) -> np.ndarray:
    if adj_mx is None:
        raise ValueError("OpenCity requires adj_mx from benchmark relation data.")
    if isinstance(adj_mx, torch.Tensor):
        adj = adj_mx.detach().cpu().numpy()
    else:
        adj = np.asarray(adj_mx)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"OpenCity adj_mx must be square, got {adj.shape}.")
    return adj.astype(np.float32, copy=True)


def build_opencity_graphs(
    adj_mx: Any,
    *,
    dataset_key: str = OPENCITY_DATASET_KEY,
    lape_dim: int = 8,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build OpenCity graph dictionaries from a benchmark adjacency matrix."""
    adj = _as_numpy_adjacency(adj_mx)

    sh_mx = adj.copy()
    sh_mx[sh_mx > 0] = 1
    sh_mx[sh_mx == 0] = 511

    lap_mx = cal_lape(adj.copy()).astype(np.float32)
    if lap_mx.shape[1] < lape_dim:
        lap_mx = np.pad(
            lap_mx,
            ((0, 0), (0, lape_dim - lap_mx.shape[1])),
            mode="constant",
        )
    elif lap_mx.shape[1] > lape_dim:
        lap_mx = lap_mx[:, :lape_dim]

    degree = np.sum(adj, axis=1)
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv_sqrt[positive] = 1.0 / np.sqrt(degree[positive])
    normalized = np.eye(adj.shape[0], dtype=np.float32) + (
        inv_sqrt[:, None] * adj * inv_sqrt[None, :]
    )

    return (
        {dataset_key: torch.tensor(normalized, dtype=torch.float32)},
        {dataset_key: torch.tensor(sh_mx, dtype=torch.float32)},
        {dataset_key: torch.tensor(lap_mx, dtype=torch.float32)},
    )


def build_opencity_args(configs, dataset_key: str = OPENCITY_DATASET_KEY):
    """Construct the minimal args object expected by the vendored OpenCity class."""
    lape_dim = int(getattr(configs, "opencity_lape_dim", 8))
    adj_mx_dict, sh_mx_dict, lap_mx_dict = build_opencity_graphs(
        getattr(configs, "adj_mx", None),
        dataset_key=dataset_key,
        lape_dim=lape_dim,
    )
    return SimpleNamespace(
        adj_mx_dict=adj_mx_dict,
        sh_mx_dict=sh_mx_dict,
        lap_mx_dict=lap_mx_dict,
        input_window=OPENCITY_CONTEXT,
        output_window=OPENCITY_HORIZON,
        embed_dim=int(getattr(configs, "opencity_embed_dim", 512)),
        skip_dim=int(getattr(configs, "opencity_skip_dim", 512)),
        lape_dim=lape_dim,
        geo_num_heads=int(getattr(configs, "opencity_geo_num_heads", 0)),
        sem_num_heads=int(getattr(configs, "opencity_sem_num_heads", 0)),
        tc_num_heads=int(getattr(configs, "opencity_tc_num_heads", 16)),
        t_num_heads=int(getattr(configs, "opencity_t_num_heads", 16)),
        mlp_ratio=int(getattr(configs, "opencity_mlp_ratio", 2)),
        qkv_bias=_as_bool(getattr(configs, "opencity_qkv_bias", True)),
        drop=float(getattr(configs, "opencity_drop", 0.1)),
        attn_drop=float(getattr(configs, "opencity_attn_drop", 0.3)),
        drop_path=float(getattr(configs, "opencity_drop_path", 0.0)),
        s_attn_size=int(getattr(configs, "opencity_s_attn_size", 3)),
        t_attn_size=int(getattr(configs, "opencity_t_attn_size", 1)),
        enc_depth=int(getattr(configs, "opencity_enc_depth", 6)),
        type_ln=str(getattr(configs, "opencity_type_ln", "pre")),
        type_short_path=str(getattr(configs, "opencity_type_short_path", "hop")),
        far_mask_delta=int(getattr(configs, "opencity_far_mask_delta", 5)),
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def infer_interval_minutes(timestamps: pd.Series) -> int:
    dates = pd.to_datetime(timestamps)
    if len(dates) <= 1:
        return 5
    diffs = dates.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 5
    interval = int(round(float(diffs.mode().iloc[0]) / 60.0))
    return max(interval, 1)


def build_opencity_time_slots(timestamps) -> np.ndarray:
    """Build OpenCity [minute_of_day, weekday] integer slots.

    OpenCity's native METR_LA preprocessing assigns 00:00 to minute slot 5
    for five-minute data, and weekdays are one-based.
    """
    dates = pd.to_datetime(np.asarray(timestamps).reshape(-1))
    interval = infer_interval_minutes(pd.Series(dates))
    minute = dates.hour * 60 + dates.minute + interval
    minute = ((minute - 1) % 1440) + 1
    weekday = dates.dayofweek + 1
    slots = np.stack([minute, weekday], axis=-1).astype(np.float32)
    return slots.reshape(np.asarray(timestamps).shape + (2,))


def shared_time_mark(mark: torch.Tensor) -> torch.Tensor:
    if mark.ndim == 3:
        return mark
    if mark.ndim == 4:
        return mark[:, 0, :, :]
    raise ValueError(
        "OpenCity expects time marks with shape [B,T,2] or [B,N,T,2], "
        f"got {tuple(mark.shape)}."
    )


def expand_time_mark(mark: torch.Tensor, num_nodes: int) -> torch.Tensor:
    mark = shared_time_mark(mark)
    return mark.unsqueeze(2).expand(-1, -1, num_nodes, -1)


def build_opencity_inputs(
    history: torch.Tensor,
    target: torch.Tensor,
    input_mark: torch.Tensor,
    target_mark: torch.Tensor,
    *,
    context_len: int = OPENCITY_CONTEXT,
    output_len: int = OPENCITY_HORIZON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert benchmark tensors to OpenCity's fixed 288->288 tensors."""
    if history.ndim != 4:
        raise ValueError(
            "OpenCity expects benchmark spatial history [B,N,T,C], "
            f"got {tuple(history.shape)}."
        )
    if history.shape[-1] != 1:
        raise ValueError(
            "OpenCity benchmark integration currently expects one target channel, "
            f"got {history.shape[-1]} channels."
        )
    if history.shape[2] < context_len:
        raise ValueError(
            f"OpenCity needs at least {context_len} history steps, got {history.shape[2]}."
        )

    batch_size, num_nodes = history.shape[0], history.shape[1]
    x_value = history[:, :, -context_len:, :1].permute(0, 2, 1, 3).contiguous()
    x_mark = expand_time_mark(input_mark, num_nodes)[:, -context_len:, :, :].to(
        device=history.device,
        dtype=history.dtype,
    )
    x_open = torch.cat([x_value, x_mark], dim=-1)

    y_mark = expand_time_mark(target_mark, num_nodes)[:, -output_len:, :, :].to(
        device=history.device,
        dtype=history.dtype,
    )
    if y_mark.shape[1] != output_len:
        raise ValueError(
            f"OpenCity needs {output_len} future time steps, got {y_mark.shape[1]}."
        )

    if target is not None and target.ndim == 4 and target.shape[2] >= output_len:
        y_value = target[:, :, -output_len:, :1].permute(0, 2, 1, 3).contiguous()
    else:
        y_value = torch.zeros(
            batch_size,
            output_len,
            num_nodes,
            1,
            device=history.device,
            dtype=history.dtype,
        )
    y_open = torch.cat([y_value, y_mark], dim=-1)
    return x_open, y_open, y_value


def restore_opencity_output(output: torch.Tensor, horizon: int | None = None) -> torch.Tensor:
    """Convert OpenCity [B,T,N,1] output to benchmark [B,N,T,1]."""
    if output.ndim != 4:
        raise ValueError(f"OpenCity raw output must be [B,T,N,1], got {tuple(output.shape)}.")
    if horizon is not None:
        output = output[:, : int(horizon), :, :]
    return output.permute(0, 2, 1, 3).contiguous()
