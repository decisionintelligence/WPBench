# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "wape",
    "msmape",
    "mae_norm",
    "mse_norm",
    "rmse_norm",
    "mape_norm",
    "smape_norm",
    "mase_norm",
    "wape_norm",
    "msmape_norm",
]


# ---------- Base error functions ----------

def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return (actual - predicted) / actual


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


def mase(actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)
    dif = sum(abs(hist_data[i] - hist_data[i - seasonality]) for i in range((seasonality + 1), len(hist_data)))
    scale = scale * dif
    return (sum(abs(actual - predicted)) / scale)[0]


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

def _flatten_for_scaler(actual: np.ndarray, predicted: np.ndarray):
    """Helper to flatten input to (samples, C) for scaler.transform()."""
    if actual.ndim == 4:  # (B, N, T, C)
        B, N, T, C = actual.shape
        return actual.reshape(B * N * T, C), predicted.reshape(B * N * T, C)
    elif actual.ndim == 3:  # (B, T, C)
        B, T, C = actual.shape
        return actual.reshape(B * T, C), predicted.reshape(B * T, C)
    elif actual.ndim == 2:
        return actual, predicted
    else:
        raise ValueError("Unsupported input shape for normalization.")


def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted)
    return scaler.transform(actual_flat) - scaler.transform(predicted_flat)


def _percentage_error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted)
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
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted)
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
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted)
    actual_scaled = scaler.transform(actual_flat)
    predicted_scaled = scaler.transform(predicted_flat)
    smape = 2.0 * np.abs(actual_scaled - predicted_scaled) / (np.abs(actual_scaled) + np.abs(predicted_scaled))
    return np.mean(smape) * 100


def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    if actual.ndim == 4:
        B, N = actual.shape[:2]
        wape_list = []
        a_flat, p_flat = _flatten_for_scaler(actual, predicted)
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
            a_flat, p_flat = _flatten_for_scaler(actual[b], predicted[b])
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
    actual_flat, predicted_flat = _flatten_for_scaler(actual, predicted)
    actual_scaled = scaler.transform(actual_flat)
    predicted_scaled = scaler.transform(predicted_flat)
    comparator = np.full_like(actual_scaled, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted_scaled) + np.abs(actual_scaled) + epsilon)
    return np.mean(2 * np.abs(predicted_scaled - actual_scaled) / denom) * 100
