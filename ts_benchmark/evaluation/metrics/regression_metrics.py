# -*- coding: utf-8 -*-

import numpy as np

DEFAULT_METRICS = [
    "mae",
    "mse",
    "rmse",
    "nmae",
    "nmse",
    "nrmse",
    "nmbe",
    "maape",
    "mape",
    "smape",
    "mase",
    "rmsse",
    "wape",
    "msmape",
    "edf",
    "opencity_mae",
    "opencity_mse",
    "opencity_rmse",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
    # "ndtw",
    # "nddtw",
]

__all__ = DEFAULT_METRICS + [
    "dtw",
    "ndtw",
    "nddtw",
]

_EPSILON = 1e-8


# ---------- Base error functions ----------

def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return (actual - predicted) / actual


def _safe_denominator(values: np.ndarray, epsilon: float = _EPSILON) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < epsilon, epsilon, values)


def _reshape_norm_factor(
    norm_factor: np.ndarray,
    actual: np.ndarray,
    epsilon: float = _EPSILON,
) -> np.ndarray:
    actual = np.asarray(actual)
    factor = np.asarray(norm_factor, dtype=float)
    factor = np.maximum(np.abs(factor), epsilon)

    if factor.ndim == 0 or factor.size == 1:
        return float(factor.reshape(-1)[0])

    if actual.ndim == 4:
        _, node_count, _, channel_count = actual.shape
        if factor.shape == (node_count, channel_count):
            return factor.reshape(1, node_count, 1, channel_count)
        if factor.ndim == 1 and factor.size == node_count * channel_count:
            return factor.reshape(1, node_count, 1, channel_count)
        raise ValueError(
            "norm_factor for 4D actual must be scalar, (N, C), or length N*C"
        )
    elif actual.ndim == 3:
        channel_count = actual.shape[-1]
        if factor.size == channel_count:
            return factor.reshape(1, 1, channel_count)
        raise ValueError("norm_factor for 3D actual must be scalar or length C")
    elif actual.ndim == 2:
        channel_count = actual.shape[-1]
        if factor.size == channel_count:
            return factor.reshape(1, channel_count)
        raise ValueError("norm_factor for 2D actual must be scalar or length C")

    return factor


def _get_norm_factor(
    actual: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
) -> np.ndarray:
    if norm_factor is None:
        raise ValueError(
            "Capacity-normalized metrics require explicit norm_factor."
        )
    return _reshape_norm_factor(norm_factor, actual, epsilon)


def _flatten_series(actual: np.ndarray, predicted: np.ndarray):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted must have the same shape")

    if actual.ndim == 1:
        return actual.reshape(1, -1), predicted.reshape(1, -1)
    if actual.ndim == 2:  # (T, C)
        return actual.T, predicted.T
    if actual.ndim == 3:  # (B, T, C)
        return (
            actual.transpose(0, 2, 1).reshape(-1, actual.shape[1]),
            predicted.transpose(0, 2, 1).reshape(-1, predicted.shape[1]),
        )
    if actual.ndim == 4:  # (B, N, T, C)
        return (
            actual.transpose(0, 1, 3, 2).reshape(-1, actual.shape[2]),
            predicted.transpose(0, 1, 3, 2).reshape(-1, predicted.shape[2]),
        )
    raise ValueError("Unsupported input shape for series-wise metrics")


def _require_fastdtw():
    try:
        from fastdtw import fastdtw
    except ImportError as exc:
        raise ImportError(
            "ndtw/nddtw requires fastdtw. Please install it with `pip install fastdtw`."
        ) from exc
    return fastdtw


def _manhattan_distance(x, y) -> float:
    return float(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)).sum())


def _fastdtw_mean_distance(
    actual_series: np.ndarray,
    predicted_series: np.ndarray,
) -> float:
    fastdtw = _require_fastdtw()
    distances = []
    for i in range(actual_series.shape[0]):
        d, _ = fastdtw(
            actual_series[i].reshape(-1, 1),
            predicted_series[i].reshape(-1, 1),
            dist=_manhattan_distance,
        )
        distances.append(d)
    return float(np.mean(distances))


def _compute_derivative_1d(time_series: np.ndarray) -> np.ndarray:
    time_series = np.asarray(time_series, dtype=float)
    if len(time_series) < 2:
        return np.zeros_like(time_series, dtype=float)
    if len(time_series) == 2:
        diff = time_series[1] - time_series[0]
        return np.array([diff, diff], dtype=float)

    derivative = np.zeros_like(time_series, dtype=float)
    for t in range(1, len(time_series) - 1):
        derivative[t] = (
            (time_series[t] - time_series[t - 1])
            + ((time_series[t + 1] - time_series[t - 1]) / 2.0)
        ) / 2.0
    derivative[0] = derivative[1]
    derivative[-1] = derivative[-2]
    return derivative


def _scaled_error(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int,
    squared: bool = False,
    epsilon: float = _EPSILON,
) -> float:
    if hist_data is None:
        raise ValueError("hist_data is required for scaled error metrics")
    if seasonality <= 0:
        raise ValueError("seasonality must be a positive integer")

    hist = np.asarray(hist_data, dtype=float)
    if hist.shape[0] <= seasonality:
        return np.nan

    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    naive_errors = hist[seasonality:] - hist[:-seasonality]
    if squared:
        numerator = np.mean(np.square(errors))
        denominator = np.mean(np.square(naive_errors))
    else:
        numerator = np.mean(np.abs(errors))
        denominator = np.mean(np.abs(naive_errors))

    if np.abs(denominator) < epsilon:
        return np.nan
    return numerator / denominator


# ---------- Basic metrics ----------

def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    if actual.ndim == 2:
        return np.sqrt(mse(actual, predicted))
    elif actual.ndim == 3:
        B = actual.shape[0]
        rmse_list = [np.sqrt(mse(actual[b], predicted[b])) for b in range(B)]
        return np.mean(np.stack(rmse_list), axis=0).tolist()
    elif actual.ndim == 4:
        return np.sqrt(mse(actual, predicted))
    else:
        raise ValueError("Unsupported input shape for rmse")


def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_error(actual, predicted)))


def nmae(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    factor = _get_norm_factor(actual, norm_factor, hist_data, epsilon)
    return np.mean(np.abs(_error(actual, predicted)) / factor)


def nmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    factor = _get_norm_factor(actual, norm_factor, hist_data, epsilon)
    return np.mean(np.square(_error(actual, predicted) / factor))


def nrmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    return np.sqrt(
        nmse(
            actual,
            predicted,
            norm_factor=norm_factor,
            hist_data=hist_data,
            epsilon=epsilon,
        )
    )


def nmbe(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    factor = _get_norm_factor(actual, norm_factor, hist_data, epsilon)
    return np.mean(_error(actual, predicted) / factor)


def _opencity_masked_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
    mae_thresh: float = 0.0,
):
    mask = actual > mae_thresh
    return actual[mask], predicted[mask]


def opencity_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    mae_thresh: float = 0.0,
    **kwargs,
):
    actual_masked, predicted_masked = _opencity_masked_arrays(
        actual, predicted, mae_thresh
    )
    return np.mean(np.abs(actual_masked - predicted_masked))


def opencity_mse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mae_thresh: float = 0.0,
    **kwargs,
):
    actual_masked, predicted_masked = _opencity_masked_arrays(
        actual, predicted, mae_thresh
    )
    return np.mean(np.square(actual_masked - predicted_masked))


def opencity_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    mae_thresh: float = 0.0,
    **kwargs,
):
    return np.sqrt(opencity_mse(actual, predicted, mae_thresh=mae_thresh))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs,
):
    return _scaled_error(actual, predicted, hist_data, seasonality, squared=False)


def rmsse(
    actual: np.ndarray,
    predicted: np.ndarray,
    hist_data: np.ndarray,
    seasonality: int = 1,
    **kwargs,
):
    scaled = _scaled_error(actual, predicted, hist_data, seasonality, squared=True)
    return np.sqrt(scaled)


def maape(
    actual: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = _EPSILON,
    **kwargs,
):
    percentage_error = np.abs(_error(actual, predicted) / _safe_denominator(actual, epsilon))
    return np.mean(np.arctan(percentage_error))


def edf(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    abs_error = nmae(
        actual,
        predicted,
        norm_factor=norm_factor,
        hist_data=hist_data,
        epsilon=epsilon,
    )
    if abs_error < epsilon:
        return np.nan
    root_error = nrmse(
        actual,
        predicted,
        norm_factor=norm_factor,
        hist_data=hist_data,
        epsilon=epsilon,
    )
    return root_error / abs_error


def _dtw_1d(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rows, cols = len(actual), len(predicted)
    distances = np.full((rows + 1, cols + 1), np.inf)
    distances[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = (actual[i - 1] - predicted[j - 1]) ** 2
            distances[i, j] = cost + min(
                distances[i - 1, j],
                distances[i, j - 1],
                distances[i - 1, j - 1],
            )
    return np.sqrt(distances[rows, cols])


def dtw(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    actual_series, predicted_series = _flatten_series(actual, predicted)
    distances = [
        _dtw_1d(actual_series[i], predicted_series[i])
        for i in range(actual_series.shape[0])
    ]
    return np.mean(distances)


def ndtw(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    """Capacity-normalized Dynamic Time Warping; smaller is better."""
    factor = _get_norm_factor(actual, norm_factor, hist_data, epsilon)
    actual_norm = np.asarray(actual, dtype=float) / factor
    predicted_norm = np.asarray(predicted, dtype=float) / factor
    actual_series, predicted_series = _flatten_series(actual_norm, predicted_norm)
    return _fastdtw_mean_distance(actual_series, predicted_series)


def nddtw(
    actual: np.ndarray,
    predicted: np.ndarray,
    norm_factor: np.ndarray = None,
    hist_data: np.ndarray = None,
    epsilon: float = _EPSILON,
    **kwargs,
):
    """Capacity-normalized Derivative Dynamic Time Warping; smaller is better."""
    factor = _get_norm_factor(actual, norm_factor, hist_data, epsilon)
    actual_norm = np.asarray(actual, dtype=float) / factor
    predicted_norm = np.asarray(predicted, dtype=float) / factor
    actual_series, predicted_series = _flatten_series(actual_norm, predicted_norm)
    actual_derivatives = np.array(
        [_compute_derivative_1d(series) for series in actual_series]
    )
    predicted_derivatives = np.array(
        [_compute_derivative_1d(series) for series in predicted_series]
    )
    return _fastdtw_mean_distance(actual_derivatives, predicted_derivatives)


def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100


def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100


def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    if actual.ndim == 2:
        return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    elif actual.ndim == 3:
        B = actual.shape[0]
        wape_list = []
        for b in range(B):
            numerator = np.sum(np.abs(actual[b] - predicted[b]))
            denominator = np.sum(np.abs(actual[b]))
            wape_b = (numerator / denominator) * 100
            wape_list.append(wape_b)
        return np.mean(np.stack(wape_list), axis=0).tolist()
    elif actual.ndim == 4:
       return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100
    else:
        raise ValueError("Unsupported input shape for wape")


def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    return np.mean(2 * np.abs(predicted - actual) / denom) * 100


# ---------- Normalized versions ----------

def _scaler_feature_count(scaler: object) -> int:
    if hasattr(scaler, "n_features_in_"):
        return int(scaler.n_features_in_)
    if hasattr(scaler, "mean_"):
        return int(np.asarray(scaler.mean_).shape[0])
    return -1


def _flatten_for_scaler(actual: np.ndarray, predicted: np.ndarray, scaler: object = None):
    """Helper to flatten input to (samples, C) for scaler.transform()."""
    if actual.ndim == 4:  # (B, N, T, C)
        B, N, T, C = actual.shape
        if scaler is not None and _scaler_feature_count(scaler) == N * C:
            return (
                actual.transpose(0, 2, 1, 3).reshape(B * T, N * C),
                predicted.transpose(0, 2, 1, 3).reshape(B * T, N * C),
            )
        return actual.reshape(B * N * T, C), predicted.reshape(B * N * T, C)
    elif actual.ndim == 3:  # (B, T, C)
        B, T, C = actual.shape
        return actual.reshape(B * T, C), predicted.reshape(B * T, C)
    elif actual.ndim == 2:
        return actual, predicted
    else:
        raise ValueError("Unsupported input shape for normalization.")


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted, scaler)
    return scaler.transform(actual_flat) - scaler.transform(predicted_flat)


def _percentage_error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted, scaler)
    return (scaler.transform(actual_flat) - scaler.transform(predicted_flat)) / scaler.transform(actual_flat)


def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    if actual.ndim == 2:
        return np.sqrt(mse_norm(actual, predicted, scaler))
    elif actual.ndim == 3:
        B = actual.shape[0]
        rmse_list = [np.sqrt(mse_norm(actual[b], predicted[b], scaler)) for b in range(B)]
        return np.mean(np.stack(rmse_list), axis=0).tolist()
    elif actual.ndim == 4:
        return np.sqrt(mse_norm(actual, predicted, scaler))
    else:
        raise ValueError("Unsupported input shape for rmse_norm")


def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def mase_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted, scaler)
    hist_flat = hist_data.reshape(-1, hist_data.shape[-1])
    actual_scaled = scaler.transform(actual_flat)
    predicted_scaled = scaler.transform(predicted_flat)
    hist_scaled = scaler.transform(hist_flat)

    if seasonality == 2:
        return -1

    scale = len(predicted_scaled) / (len(hist_scaled) - seasonality)
    dif = sum(abs(hist_scaled[i] - hist_scaled[i - seasonality]) for i in range(seasonality + 1, len(hist_scaled)))
    scale = scale * dif

    return (sum(abs(actual_scaled - predicted_scaled)) / scale)[0]


def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100


def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted, scaler)
    actual_scaled = scaler.transform(actual_flat)
    predicted_scaled = scaler.transform(predicted_flat)
    smape = 2.0 * np.abs(actual_scaled - predicted_scaled) / (np.abs(actual_scaled) + np.abs(predicted_scaled))
    return np.mean(smape) * 100


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    if actual.ndim == 4:
        B, N = actual.shape[:2]
        wape_list = []
        a_flat, p_flat = _flatten_for_scaler(actual, predicted, scaler)
        a_scaled = scaler.transform(a_flat)
        p_scaled = scaler.transform(p_flat)
        numerator = np.sum(np.abs(a_scaled - p_scaled))
        denominator = np.sum(np.abs(a_scaled))
        wape_bn = (numerator / denominator) * 100
        return wape_bn
    elif actual.ndim == 3:
        B = actual.shape[0]
        wape_list = []
        for b in range(B):
            a_flat, p_flat = _flatten_for_scaler(actual[b], predicted[b], scaler)
            a_scaled = scaler.transform(a_flat)
            p_scaled = scaler.transform(p_flat)
            numerator = np.sum(np.abs(a_scaled - p_scaled))
            denominator = np.sum(np.abs(a_scaled))
            wape_b = (numerator / denominator) * 100
            wape_list.append(wape_b)
        return np.mean(np.stack(wape_list), axis=0).tolist()
    else:
        a_scaled, p_scaled = scaler.transform(actual), scaler.transform(predicted)
        numerator = np.sum(np.abs(a_scaled - p_scaled))
        denominator = np.sum(np.abs(a_scaled))
        return (numerator / denominator) * 100


def msmape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, epsilon: float = 0.1, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted, scaler)
    actual_scaled = scaler.transform(actual_flat)
    predicted_scaled = scaler.transform(predicted_flat)
    comparator = np.full_like(actual_scaled, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted_scaled) + np.abs(actual_scaled) + epsilon)
    return np.mean(2 * np.abs(predicted_scaled - actual_scaled) / denom) * 100
