"""Tensor layout bridge for FactoST STA inside the benchmark."""

from __future__ import annotations

import torch

FACTOST_BASE_SLOT_WIDTH = 5


def _shared_time_mark(mark: torch.Tensor) -> torch.Tensor:
    if mark.ndim == 3:
        return mark
    if mark.ndim == 4:
        return mark[:, 0, :, :]
    raise ValueError(
        "FactoST STA expects temporal marks with shape [B, T, 5] or [B, N, T, 5], "
        f"got {tuple(mark.shape)}."
    )


def reorder_target_only_input(x: torch.Tensor) -> torch.Tensor:
    """Convert benchmark spatial input [B, N, T, 1] to [B, T, N, 1]."""
    if x.ndim != 4:
        raise ValueError(
            f"FactoST STA expects spatial input with shape [B, N, T, 1], got {tuple(x.shape)}."
        )
    if x.shape[-1] != 1:
        raise ValueError(
            "FactoST STA v1 expects target-only input with one channel. "
            f"Got {x.shape[-1]} channels; select the target variable before this adapter."
        )
    return x.permute(0, 2, 1, 3).contiguous()


def build_factost_sta_inputs(
    x_target_only: torch.Tensor,
    input_mark: torch.Tensor,
    target_mark: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FactoST history input and future temporal features."""
    x_values = reorder_target_only_input(x_target_only)
    batch_size, history_len, num_nodes, _ = x_values.shape

    input_slots = _shared_time_mark(input_mark).to(
        device=x_values.device,
        dtype=x_values.dtype,
    )
    if input_slots.shape != (batch_size, history_len, FACTOST_BASE_SLOT_WIDTH):
        raise ValueError(
            "FactoST STA input_mark must resolve to [B, T, 5]. "
            f"Expected {(batch_size, history_len, FACTOST_BASE_SLOT_WIDTH)}, got {tuple(input_slots.shape)}."
        )

    input_slots = input_slots.unsqueeze(2).expand(-1, -1, num_nodes, -1)
    x_factost = torch.cat([x_values, input_slots], dim=-1)

    future_slots = _shared_time_mark(target_mark)[:, -int(horizon) :, :].to(
        device=x_values.device,
        dtype=x_values.dtype,
    )
    if future_slots.shape != (batch_size, int(horizon), FACTOST_BASE_SLOT_WIDTH):
        raise ValueError(
            "FactoST STA target_mark must resolve to [B, label_len + H, 5]. "
            f"Expected future shape {(batch_size, int(horizon), FACTOST_BASE_SLOT_WIDTH)}, "
            f"got {tuple(future_slots.shape)}."
        )

    return x_factost, future_slots


def restore_factost_sta_output(output: torch.Tensor) -> torch.Tensor:
    """Convert vendored FactoST output [B, H, N] to benchmark [B, N, H, 1]."""
    if output.ndim == 3:
        return output.permute(0, 2, 1).unsqueeze(-1).contiguous()
    if output.ndim == 4:
        return output
    raise ValueError(
        f"FactoST STA output must have shape [B, H, N] or [B, N, H, C], got {tuple(output.shape)}."
    )
