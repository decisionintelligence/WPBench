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

def _error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return actual - predicted

def _percentage_error(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return (actual - predicted) / actual

def mse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.square(_error(actual, predicted)))

def rmse(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.sqrt(mse(actual, predicted))

def mae(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_error(actual, predicted)))

def mase(actual: np.ndarray, predicted: np.ndarray, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
    if seasonality == 2:
        return -1
    scale = len(predicted) / (len(hist_data) - seasonality)
    dif = 0
    for i in range((seasonality + 1), len(hist_data)):
        dif += abs(hist_data[i] - hist_data[i - seasonality])
    scale = scale * dif
    return (sum(abs(actual - predicted)) / scale)[0]

def mape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(np.abs(_percentage_error(actual, predicted))) * 100

def smape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100

def wape(actual: np.ndarray, predicted: np.ndarray, **kwargs):
    return np.sum(np.abs(actual - predicted)) / np.sum(np.abs(actual)) * 100

def msmape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 0.1, **kwargs):
    comparator = np.full_like(actual, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted) + np.abs(actual) + epsilon)
    return np.mean(2 * np.abs(predicted - actual) / denom) * 100

# ---------- Normalized versions ----------

def _error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return scaler.transform(actual) - scaler.transform(predicted)

def _percentage_error_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return (scaler.transform(actual) - scaler.transform(predicted)) / scaler.transform(actual)

def mse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))

def rmse_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.sqrt(mse_norm(actual, predicted, scaler))

def mae_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))

def mase_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, hist_data: np.ndarray, seasonality: int = 2, **kwargs):
    is_3d = actual.ndim == 3
    if is_3d:
        N, T, C = actual.shape
        actual_2d = actual.reshape(N * T, C)
        predicted_2d = predicted.reshape(N * T, C)
        hist_2d = hist_data.reshape(-1, hist_data.shape[-1])
        actual_scaled = scaler.transform(actual_2d)
        predicted_scaled = scaler.transform(predicted_2d)
        hist_scaled = scaler.transform(hist_2d)
    else:
        actual_scaled = scaler.transform(actual)
        predicted_scaled = scaler.transform(predicted)
        hist_scaled = scaler.transform(hist_data)

    if seasonality == 2:
        return -1

    scale = len(predicted_scaled) / (len(hist_scaled) - seasonality)
    dif = 0
    for i in range((seasonality + 1), len(hist_scaled)):
        dif += abs(hist_scaled[i] - hist_scaled[i - seasonality])
    scale = scale * dif

    return (sum(abs(actual_scaled - predicted_scaled)) / scale)[0]

def mape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    return np.mean(np.abs(_percentage_error_norm(actual, predicted, scaler))) * 100

def smape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    is_3d = actual.ndim == 3
    if is_3d:
        N, T, C = actual.shape
        actual_2d = actual.reshape(N * T, C)
        predicted_2d = predicted.reshape(N * T, C)
        actual_scaled = scaler.transform(actual_2d)
        predicted_scaled = scaler.transform(predicted_2d)
    else:
        actual_scaled = scaler.transform(actual)
        predicted_scaled = scaler.transform(predicted)

    smape = 2.0 * np.abs(actual_scaled - predicted_scaled) / (np.abs(actual_scaled) + np.abs(predicted_scaled))
    if is_3d:
        smape = smape.reshape(N, T, C)
    return np.mean(smape) * 100

def wape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, **kwargs):
    is_3d = actual.ndim == 3
    if is_3d:
        N, T, C = actual.shape
        actual_2d = actual.reshape(N * T, C)
        predicted_2d = predicted.reshape(N * T, C)
        actual_scaled = scaler.transform(actual_2d)
        predicted_scaled = scaler.transform(predicted_2d)
    else:
        actual_scaled = scaler.transform(actual)
        predicted_scaled = scaler.transform(predicted)

    numerator = np.sum(np.abs(actual_scaled - predicted_scaled))
    denominator = np.sum(np.abs(actual_scaled))
    return (numerator / denominator) * 100

def msmape_norm(actual: np.ndarray, predicted: np.ndarray, scaler: object, epsilon: float = 0.1, **kwargs):
    is_3d = actual.ndim == 3
    if is_3d:
        N, T, C = actual.shape
        actual_2d = actual.reshape(N * T, C)
        predicted_2d = predicted.reshape(N * T, C)
        actual_scaled = scaler.transform(actual_2d)
        predicted_scaled = scaler.transform(predicted_2d)
    else:
        actual_scaled = scaler.transform(actual)
        predicted_scaled = scaler.transform(predicted)

    comparator = np.full_like(actual_scaled, 0.5 + epsilon)
    denom = np.maximum(comparator, np.abs(predicted_scaled) + np.abs(actual_scaled) + epsilon)
    msmape_per_series = np.mean(2 * np.abs(predicted_scaled - actual_scaled) / denom) * 100
    return msmape_per_series
