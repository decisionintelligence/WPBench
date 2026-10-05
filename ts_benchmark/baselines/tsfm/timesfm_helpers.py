from typing import Sequence

import numpy as np
import pandas as pd
from einops import rearrange
from ts_benchmark.baselines.tsfm.submodules.timesfm.timesfm_base import freq_map


def preprocess_timesfm_inputs(
    inputs: Sequence[np.ndarray],
    freq: Sequence[int],
    *,
    context_len: int,
    horizon_len: int,
    global_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pad / truncate contexts like ``TimesFmBase._preprocess`` (for finetune decoder inference)."""
    input_ts, input_padding, inp_freq = [], [], []
    pmap_pad = ((len(inputs) - 1) // global_batch_size + 1) * global_batch_size - len(
        inputs
    )
    for i, ts in enumerate(inputs):
        ts = np.asarray(ts, dtype=np.float32)
        input_len = int(ts.shape[0])
        padding = np.zeros(shape=(input_len + horizon_len,), dtype=np.float32)
        if input_len < context_len:
            num_front_pad = context_len - input_len
            ts = np.concatenate(
                [np.zeros(shape=(num_front_pad,), dtype=np.float32), ts], axis=0
            )
            padding = np.concatenate(
                [np.ones(shape=(num_front_pad,), dtype=np.float32), padding], axis=0
            )
        elif input_len > context_len:
            ts = ts[-context_len:]
            padding = padding[-(context_len + horizon_len) :]

        input_ts.append(ts)
        input_padding.append(padding)
        inp_freq.append(int(freq[i]))

    for _ in range(pmap_pad):
        input_ts.append(input_ts[-1])
        input_padding.append(input_padding[-1])
        inp_freq.append(inp_freq[-1])

    return (
        np.stack(input_ts, axis=0),
        np.stack(input_padding, axis=0),
        np.array(inp_freq, dtype=np.int32).reshape(-1, 1),
        pmap_pad,
    )

def infer_timesfm_freq_type(index, fallback="h"):
    freq = pd.infer_freq(index)
    return freq_map(freq or fallback)

def wide_to_timesfm_list(history_2d: np.ndarray):
    return [history_2d[:, i].astype(np.float32) for i in range(history_2d.shape[1])]

def timesfm_forecast_to_wide(mean_forecast: np.ndarray):
    return mean_forecast.T.astype(np.float32)

def batch_wide_to_timesfm_list(batch_3d: np.ndarray):
    b, t, m = batch_3d.shape
    flat = rearrange(batch_3d, "b t m -> (b m) t")
    return [flat[i].astype(np.float32) for i in range(flat.shape[0])]

def batch_forecast_back(mean_forecast: np.ndarray, batch_size: int, total_vars: int):
    return rearrange(mean_forecast, "(b m) h -> b h m", b=batch_size, m=total_vars)
