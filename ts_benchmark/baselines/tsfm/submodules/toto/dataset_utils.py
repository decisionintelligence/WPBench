from functools import reduce
from typing import NamedTuple, Optional

import numpy as np
import torch
from einops import repeat


def pad_array(values: torch.Tensor, patch_stride: int) -> torch.Tensor:
    if isinstance(values, np.ndarray):
        values = torch.from_numpy(values)
    series_len = values.shape[-1]
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    if values.ndim == 2:
        padded_values = torch.zeros((values.shape[0], padded_length), dtype=values.dtype, device=values.device)
    elif values.ndim == 3:
        padded_values = torch.zeros(
            (values.shape[0], values.shape[1], padded_length),
            dtype=values.dtype,
            device=values.device,
        )
    else:
        raise ValueError(f"Unsupported number of dimensions: {values.ndim}")
    padded_values[..., -series_len:] = values
    return padded_values


def pad_id_mask(id_mask: torch.Tensor, patch_stride: int) -> torch.Tensor:
    series_len = id_mask.shape[-1]
    padded_length = int(np.ceil(series_len / patch_stride) * patch_stride)
    padding_amount = padded_length - series_len
    left_edge = id_mask[..., 0]
    if id_mask.ndim == 2:
        padding = repeat(
            left_edge,
            "variates -> variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=1)
    elif id_mask.ndim == 3:
        padding = repeat(
            left_edge,
            "batch variates -> batch variates padding_amount",
            padding_amount=padding_amount,
        )
        id_mask = torch.cat([padding, id_mask], dim=2)
    else:
        raise ValueError(f"Unsupported number of dimensions: {id_mask.ndim}")
    return id_mask


def is_extreme_value(t: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(t):
        max_value = torch.finfo(t.dtype).max
    else:
        max_value = torch.iinfo(t.dtype).max
    return reduce(
        torch.logical_or,
        (
            torch.isinf(t),
            torch.isnan(t),
            t.abs() >= max_value / 2,
        ),
    )


def replace_extreme_values(t: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.where(is_extreme_value(t), torch.tensor(replacement, dtype=t.dtype, device=t.device), t)


class MaskedTimeseries(NamedTuple):
    series: torch.Tensor
    padding_mask: torch.Tensor
    id_mask: Optional[torch.Tensor]
    timestamp_seconds: torch.Tensor
    time_interval_seconds: torch.Tensor
    num_exogenous_variables: int = 0


class CausalMaskedTimeseries(NamedTuple):
    series: torch.Tensor
    padding_mask: torch.Tensor
    id_mask: Optional[torch.Tensor]
    timestamp_seconds: torch.Tensor
    time_interval_seconds: torch.Tensor
    input_slice: slice
    target_slice: slice
    num_exogenous_variables: int = 0
