from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from ts_benchmark.baselines.tsfm.submodules.toto.dataset_utils import MaskedTimeseries


def bench_freq_to_seconds(freq: str | None) -> int:
    if not freq:
        return 3600
    try:
        offset = pd.tseries.frequencies.to_offset(str(freq).strip())
    except (ValueError, TypeError, AttributeError):
        return 3600

    try:
        return max(1, int(offset.nanos // 1_000_000_000))
    except ValueError:
        if isinstance(offset, pd.offsets.Week):
            return int(offset.n * 7 * 24 * 60 * 60)
        if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
            return 30 * 24 * 60 * 60
        if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
            return 90 * 24 * 60 * 60
        if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
            return int(365.25 * 24 * 60 * 60)
        return 3600


def build_masked_timeseries(
    input_btc: torch.Tensor,
    freq: str | None,
    patch_stride: int,
    valid_mask_bt: Optional[torch.Tensor] = None,
    timestamp_seconds_bt: Optional[torch.Tensor] = None,
    last_observed_timestamp_seconds_b: Optional[torch.Tensor] = None,
    time_interval_seconds_b: Optional[torch.Tensor] = None,
) -> MaskedTimeseries:
    del patch_stride
    if input_btc.dim() != 3:
        raise ValueError(
            f"build_masked_timeseries expects input shape [B,T,C], got {tuple(input_btc.shape)}"
        )

    interval_sec = bench_freq_to_seconds(freq)
    input_bvt = input_btc.permute(0, 2, 1).contiguous()
    series = input_bvt
    b, v, t = input_bvt.shape

    if valid_mask_bt is None:
        padding_mask = torch.ones_like(input_bvt, dtype=torch.bool)
    else:
        if valid_mask_bt.dim() != 2 or tuple(valid_mask_bt.shape) != (b, t):
            raise ValueError(
                "valid_mask_bt must have shape [B,T] matching input_btc; "
                f"got {tuple(valid_mask_bt.shape)} for input {tuple(input_btc.shape)}"
            )
        padding_mask = (
            valid_mask_bt.to(device=input_bvt.device, dtype=torch.bool)
            .unsqueeze(1)
            .expand(b, v, t)
            .contiguous()
        )

    id_mask = torch.zeros(
        (b, v, t),
        dtype=torch.int,
        device=input_bvt.device,
    )
    if time_interval_seconds_b is not None:
        if time_interval_seconds_b.dim() != 1 or tuple(time_interval_seconds_b.shape) != (b,):
            raise ValueError(
                "time_interval_seconds_b must have shape [B] matching input_btc; "
                f"got {tuple(time_interval_seconds_b.shape)} for input {tuple(input_btc.shape)}"
            )
        time_interval_seconds_b = time_interval_seconds_b.to(
            device=input_bvt.device,
            dtype=torch.int64,
        ).contiguous()
    else:
        time_interval_seconds_b = torch.full(
            (b,),
            int(interval_sec),
            dtype=torch.int64,
            device=input_bvt.device,
        )

    if last_observed_timestamp_seconds_b is not None:
        if last_observed_timestamp_seconds_b.dim() != 1 or tuple(last_observed_timestamp_seconds_b.shape) != (b,):
            raise ValueError(
                "last_observed_timestamp_seconds_b must have shape [B] matching input_btc; "
                f"got {tuple(last_observed_timestamp_seconds_b.shape)} for input {tuple(input_btc.shape)}"
            )
        last_observed_timestamp_seconds_b = last_observed_timestamp_seconds_b.to(
            device=input_bvt.device,
            dtype=torch.int64,
        ).contiguous()
        # Mirror raw Toto's GluonTS predictor: it reconstructs the context timeline
        # from the last observed timestamp and then extends it forward.
        ar = torch.arange(t, dtype=torch.int64, device=input_bvt.device).view(1, -1)
        timestamp_seconds_bt = last_observed_timestamp_seconds_b.view(-1, 1) + (
            ar * time_interval_seconds_b.view(-1, 1)
        )
    elif timestamp_seconds_bt is None:
        # Keep the fallback anchored on the last context step so we mirror the
        # forecast_start-relative timeline expected by raw Toto, instead of
        # inventing a fresh 0-based timeline for every sample.
        ar = (torch.arange(t, dtype=torch.int64, device=input_bvt.device) - (t - 1)).view(1, -1)
        timestamp_seconds_bt = ar * time_interval_seconds_b.view(-1, 1)
    else:
        if timestamp_seconds_bt.dim() != 2 or tuple(timestamp_seconds_bt.shape) != (b, t):
            raise ValueError(
                "timestamp_seconds_bt must have shape [B,T] matching input_btc; "
                f"got {tuple(timestamp_seconds_bt.shape)} for input {tuple(input_btc.shape)}"
            )
        timestamp_seconds_bt = timestamp_seconds_bt.to(device=input_bvt.device, dtype=torch.int64).contiguous()

    i32_info = np.iinfo(np.int32)
    timestamp_seconds_bt = timestamp_seconds_bt.clamp(i32_info.min, i32_info.max).to(dtype=torch.int)
    timestamp_seconds = timestamp_seconds_bt.unsqueeze(1).expand(b, v, t).contiguous()
    time_interval_seconds = time_interval_seconds_b.to(dtype=torch.int).view(-1, 1).expand(b, v).contiguous()

    return MaskedTimeseries(
        series=series,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )
