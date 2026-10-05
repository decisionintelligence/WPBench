from __future__ import annotations

from typing import Optional

import numpy as np
from einops import rearrange
from sklearn.preprocessing import StandardScaler


SUPPORTED_EXTERNAL_SCALER_MODES = {
    "pooled_target",
    "variable",
    "legacy_wide",
    "node_variable",
}


def resolve_external_scaler_mode(
    raw_mode: Optional[str],
    default: str = "pooled_target",
) -> str:
    mode = str(raw_mode or default).lower()
    if mode == "variable":
        return "pooled_target"
    if mode == "legacy_wide":
        return "node_variable"
    if mode in {"pooled_target", "node_variable"}:
        return mode
    raise ValueError(
        f"Unknown external_scaler_mode: {raw_mode}. "
        f"Expected one of {sorted(SUPPORTED_EXTERNAL_SCALER_MODES)}."
    )


def _validate_tcn(values_tcn: np.ndarray) -> None:
    if values_tcn.ndim != 3:
        raise ValueError(
            "Expected spatiotemporal data with shape [T, C, N], "
            f"got {tuple(values_tcn.shape)}."
        )


def _validate_flat_bntc(values_flat_bntc: np.ndarray, series_num: int) -> None:
    if values_flat_bntc.ndim != 3:
        raise ValueError(
            "Expected flattened batch data with shape [(B*N), T, C], "
            f"got {tuple(values_flat_bntc.shape)}."
        )
    if int(series_num) <= 0 or values_flat_bntc.shape[0] % int(series_num) != 0:
        raise ValueError(
            "Flattened batch size must be divisible by series_num: "
            f"batch_nodes={values_flat_bntc.shape[0]}, series_num={series_num}."
        )


def _validate_wide(values_wide: np.ndarray, series_num: int, series_dim: int) -> None:
    expected = int(series_num) * int(series_dim)
    if values_wide.shape[-1] != expected:
        raise ValueError(
            "Wide input column count does not match series_num * series_dim: "
            f"got {values_wide.shape[-1]}, expected {expected}."
        )


def fit_spatial_external_scaler(
    scaler: StandardScaler,
    values_tcn: np.ndarray,
    *,
    mode: str,
) -> StandardScaler:
    values_tcn = np.asarray(values_tcn)
    _validate_tcn(values_tcn)
    mode = resolve_external_scaler_mode(mode)

    if mode == "node_variable":
        scaler.fit(rearrange(values_tcn, "t c n -> t (n c)"))
    else:
        scaler.fit(rearrange(values_tcn, "t c n -> (t n) c"))
    return scaler


def transform_tcn(
    values_tcn: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    inverse: bool = False,
) -> np.ndarray:
    if not norm:
        return values_tcn

    values_tcn = np.asarray(values_tcn)
    _validate_tcn(values_tcn)
    mode = resolve_external_scaler_mode(mode)
    transform = scaler.inverse_transform if inverse else scaler.transform
    steps, channels, series_num = values_tcn.shape

    if mode == "node_variable":
        wide = rearrange(values_tcn, "t c n -> t (n c)")
        scaled = transform(wide)
        return rearrange(
            scaled,
            "t (n c) -> t c n",
            t=steps,
            n=series_num,
            c=channels,
        )

    pooled = rearrange(values_tcn, "t c n -> (t n) c")
    scaled = transform(pooled)
    return rearrange(scaled, "(t n) c -> t c n", t=steps, n=series_num)


def inverse_transform_tcn(
    values_tcn: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
) -> np.ndarray:
    return transform_tcn(values_tcn, scaler, norm=norm, mode=mode, inverse=True)


def transform_flat_bntc(
    values_flat_bntc: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    inverse: bool = False,
) -> np.ndarray:
    if not norm:
        return values_flat_bntc

    values_flat_bntc = np.asarray(values_flat_bntc)
    _validate_flat_bntc(values_flat_bntc, series_num)
    mode = resolve_external_scaler_mode(mode)
    transform = scaler.inverse_transform if inverse else scaler.transform
    batch_nodes, steps, channels = values_flat_bntc.shape

    if mode == "node_variable":
        batch = batch_nodes // int(series_num)
        wide = rearrange(
            values_flat_bntc,
            "(b n) t c -> (b t) (n c)",
            b=batch,
            n=int(series_num),
        )
        scaled = transform(wide)
        return rearrange(
            scaled,
            "(b t) (n c) -> (b n) t c",
            b=batch,
            t=steps,
            n=int(series_num),
            c=channels,
        )

    pooled = rearrange(values_flat_bntc, "bn t c -> (bn t) c")
    scaled = transform(pooled)
    return rearrange(scaled, "(bn t) c -> bn t c", bn=batch_nodes, t=steps)


def transform_stssdl_exog_tcn(
    values_tcn: np.ndarray,
    target_scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
) -> np.ndarray:
    if not norm:
        return values_tcn

    values_tcn = np.asarray(values_tcn)
    _validate_tcn(values_tcn)
    if values_tcn.shape[1] != 2:
        raise ValueError(
            "STSSDL exog data must have exactly two channels: "
            f"[timeofday, label_Y], got {values_tcn.shape[1]}."
        )

    scaled = values_tcn.copy()
    scaled[:, 1:2, :] = transform_tcn(
        values_tcn[:, 1:2, :],
        target_scaler,
        norm=True,
        mode=mode,
    )
    return scaled


def transform_stssdl_exog_flat_bntc(
    values_flat_bntc: np.ndarray,
    target_scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
) -> np.ndarray:
    if not norm:
        return values_flat_bntc

    values_flat_bntc = np.asarray(values_flat_bntc)
    _validate_flat_bntc(values_flat_bntc, series_num)
    if values_flat_bntc.shape[-1] != 2:
        raise ValueError(
            "STSSDL exog data must have exactly two channels: "
            f"[timeofday, label_Y], got {values_flat_bntc.shape[-1]}."
        )

    scaled = values_flat_bntc.copy()
    scaled[:, :, 1:2] = transform_flat_bntc(
        values_flat_bntc[:, :, 1:2],
        target_scaler,
        norm=True,
        mode=mode,
        series_num=series_num,
    )
    return scaled


def inverse_transform_flat_bntc(
    values_flat_bntc: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
) -> np.ndarray:
    return transform_flat_bntc(
        values_flat_bntc,
        scaler,
        norm=norm,
        mode=mode,
        series_num=series_num,
        inverse=True,
    )


def transform_wide_2d(
    values_wide: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
    inverse: bool = False,
) -> np.ndarray:
    if not norm:
        return values_wide

    values_wide = np.asarray(values_wide)
    _validate_wide(values_wide, series_num, series_dim)
    mode = resolve_external_scaler_mode(mode)
    transform = scaler.inverse_transform if inverse else scaler.transform

    if mode == "node_variable":
        return transform(values_wide)

    steps = values_wide.shape[0]
    pooled = rearrange(
        values_wide,
        "t (n c) -> (t n) c",
        n=int(series_num),
        c=int(series_dim),
    )
    scaled = transform(pooled)
    return rearrange(
        scaled,
        "(t n) c -> t (n c)",
        t=steps,
        n=int(series_num),
    )


def inverse_transform_wide_2d(
    values_wide: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
) -> np.ndarray:
    return transform_wide_2d(
        values_wide,
        scaler,
        norm=norm,
        mode=mode,
        series_num=series_num,
        series_dim=series_dim,
        inverse=True,
    )
