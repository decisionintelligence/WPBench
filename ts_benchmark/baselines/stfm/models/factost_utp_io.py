"""Tensor layout bridge for FactoST UTP inside the benchmark."""

from __future__ import annotations

import torch


def build_factost_utp_inputs(x_target: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Convert benchmark spatial input [B, N, T, C] to UTP input [B, T, N*C]."""
    if x_target.ndim != 4:
        raise ValueError(
            "FactoST UTP expects spatial input with shape [B, N, T, C], "
            f"got {tuple(x_target.shape)}."
        )
    _, num_nodes, _, num_channels = x_target.shape
    x_utp = x_target.permute(0, 2, 1, 3).reshape(
        x_target.shape[0],
        x_target.shape[2],
        num_nodes * num_channels,
    )
    return x_utp.contiguous(), (num_nodes, num_channels)


def restore_factost_utp_output(
    output: torch.Tensor,
    layout: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert vendored UTP output [B, H, N*C] to benchmark [B, N, H, C]."""
    if output.ndim == 3:
        if layout is None:
            return output.permute(0, 2, 1).unsqueeze(-1).contiguous()
        num_nodes, num_channels = layout
        expected_vars = num_nodes * num_channels
        if output.shape[-1] != expected_vars:
            raise ValueError(
                "FactoST UTP output variable dimension does not match the original "
                f"benchmark layout: got {output.shape[-1]}, expected {expected_vars}."
            )
        return output.reshape(
            output.shape[0],
            output.shape[1],
            num_nodes,
            num_channels,
        ).permute(0, 2, 1, 3).contiguous()
    if output.ndim == 4:
        return output
    raise ValueError(
        f"FactoST UTP output must have shape [B, H, N*C] or [B, N, H, C], got {tuple(output.shape)}."
    )
