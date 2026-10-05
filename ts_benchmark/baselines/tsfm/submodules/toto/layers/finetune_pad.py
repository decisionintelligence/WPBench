# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

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
    left_edge: torch.Tensor = id_mask[..., 0]
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
