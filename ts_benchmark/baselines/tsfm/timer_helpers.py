from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.tsfm.few_shot_utils import apply_few_shot_loader
from ts_benchmark.baselines.utils import get_time_mark


SUPPORTED_TIMER_MODES = {"official_ci", "legacy_btc"}
SUPPORTED_EXTERNAL_SCALER_MODES = {"pooled_target", "legacy_wide"}


# 判断是不是合理的归一化路径
def resolve_external_scaler_mode(raw_mode: Optional[str], default: str = "pooled_target") -> str:
    mode = str(raw_mode or default).lower()
    if mode not in SUPPORTED_EXTERNAL_SCALER_MODES:
        raise ValueError(
            f"Unknown external_scaler_mode: {raw_mode}. "
            f"Expected one of {sorted(SUPPORTED_EXTERNAL_SCALER_MODES)}"
        )
    return mode


def _validate_wide_shape(values: np.ndarray, series_num: int, series_dim: int) -> None:
    if values.shape[-1] != int(series_num) * int(series_dim):
        raise ValueError(
            "Wide input column count does not match (series_num * series_dim): "
            f"got {values.shape[-1]}, expected {int(series_num) * int(series_dim)}."
        )

# 默认输入的L (N*C)的格式，先reshape成L*N*C，再(l*n)*c,F进行归一化操作/反归一化操作
def _transform_pooled_2d(
    values_2d: np.ndarray,
    scaler: StandardScaler,
    series_num: int,
    series_dim: int,
    inverse: bool,
) -> np.ndarray:
    _validate_wide_shape(values_2d, series_num, series_dim)
    steps = values_2d.shape[0]
    wide_3d = values_2d.reshape(steps, int(series_num), int(series_dim))
    pooled = wide_3d.reshape(steps * int(series_num), int(series_dim))
    pooled_scaled = scaler.inverse_transform(pooled) if inverse else scaler.transform(pooled)
    return pooled_scaled.reshape(steps, int(series_num) * int(series_dim))

# 如果出现了b维度，也这么转换，但我觉得可能用不到，存个心眼检查一下
def _transform_pooled_3d(
    values_3d: np.ndarray,
    scaler: StandardScaler,
    series_num: int,
    series_dim: int,
    inverse: bool,
) -> np.ndarray:
    _validate_wide_shape(values_3d, series_num, series_dim)
    batch, steps = values_3d.shape[0], values_3d.shape[1]
    wide_4d = values_3d.reshape(batch, steps, int(series_num), int(series_dim))
    pooled = wide_4d.reshape(batch * steps * int(series_num), int(series_dim))
    pooled_scaled = scaler.inverse_transform(pooled) if inverse else scaler.transform(pooled)
    # 在series_dim归一化后，重新还原回了b,t,n*c格式
    return pooled_scaled.reshape(batch, steps, int(series_num) * int(series_dim))


# 一种是直接每个通道*channel看作独特的，然后归一化的版本
# 一种是归一化所有的内容
# 下面的四个函数就是归一化/反归一化的接口
def transform_wide_2d(
    values_2d: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
) -> np.ndarray:
    if not norm:
        return values_2d

    mode = resolve_external_scaler_mode(mode)
    if mode == "legacy_wide":
        return scaler.transform(values_2d)
    return _transform_pooled_2d(values_2d, scaler, series_num, series_dim, inverse=False)


def inverse_transform_wide_2d(
    values_2d: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
) -> np.ndarray:
    if not norm:
        return values_2d

    mode = resolve_external_scaler_mode(mode)
    if mode == "legacy_wide":
        return scaler.inverse_transform(values_2d)
    return _transform_pooled_2d(values_2d, scaler, series_num, series_dim, inverse=True)


def transform_wide_3d(
    values_3d: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
) -> np.ndarray:
    if not norm:
        return values_3d

    mode = resolve_external_scaler_mode(mode)
    if mode == "legacy_wide":
        flat = scaler.transform(values_3d.reshape(-1, values_3d.shape[-1]))
        return flat.reshape(values_3d.shape)
    return _transform_pooled_3d(values_3d, scaler, series_num, series_dim, inverse=False)


def inverse_transform_wide_3d(
    values_3d: np.ndarray,
    scaler: StandardScaler,
    *,
    norm: bool,
    mode: str,
    series_num: int,
    series_dim: int,
) -> np.ndarray:
    if not norm:
        return values_3d

    mode = resolve_external_scaler_mode(mode)
    if mode == "legacy_wide":
        flat = scaler.inverse_transform(values_3d.reshape(-1, values_3d.shape[-1]))
        return flat.reshape(values_3d.shape)
    return _transform_pooled_3d(values_3d, scaler, series_num, series_dim, inverse=True)


# 上述4个对外归一化操作接口结束，开始拟合scaler，返回缩放后的train/valid
# 训练scaler返回归一化后的train/valid
def fit_wide_external_scaler(
    train_data: pd.DataFrame,
    valid_data: Optional[pd.DataFrame],
    *,
    norm: bool,
    scaler: Optional[StandardScaler],
    mode: str,
    series_num: int,
    series_dim: int,
) -> Tuple[StandardScaler, pd.DataFrame, Optional[pd.DataFrame]]:
    mode = resolve_external_scaler_mode(mode)
    scaler = StandardScaler() if scaler is None else scaler

    # train_np = train_data.to_numpy(dtype=np.float32, copy=True)
    train_np = train_data.to_numpy(copy=True)
    # 普通场景
    if mode == "legacy_wide":
        scaler.fit(train_np)
    else:
        _validate_wide_shape(train_np, series_num, series_dim)
        steps = train_np.shape[0]
        pooled = train_np.reshape(steps, int(series_num), int(series_dim)).reshape(
            steps * int(series_num), int(series_dim)
        )
        scaler.fit(pooled)
    
    # 如果不归一化，用的是原始数据，但是scaler还是fit了的
    if not norm:
        return scaler, train_data, valid_data

    # 懂了，scaler训练好了之后转化数据，然后返回，相当于把输入 数据，变成归一化后的数据
    train_scaled_np = transform_wide_2d(
        train_np,
        scaler,
        norm=True,
        mode=mode,
        series_num=series_num,
        series_dim=series_dim,
    )
    train_scaled = pd.DataFrame(train_scaled_np, index=train_data.index, columns=train_data.columns)

    valid_scaled = None
    if valid_data is not None:
        valid_np = valid_data.to_numpy(copy=True)
        valid_scaled_np = transform_wide_2d(
            valid_np,
            scaler,
            norm=True,
            mode=mode,
            series_num=series_num,
            series_dim=series_dim,
        )
        valid_scaled = pd.DataFrame(valid_scaled_np, index=valid_data.index, columns=valid_data.columns)

    return scaler, train_scaled, valid_scaled


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out


def infer_timer_freq(index: pd.Index, fallback: str = "h") -> str:
    try:
        dt_index = pd.to_datetime(index)
    except Exception:
        return fallback

    inferred = pd.infer_freq(dt_index)
    if inferred is None:
        return fallback
    return str(inferred).lower()


def resolve_timer_data_mode(raw_mode: str) -> str:
    mode = str(raw_mode or "official_ci").lower()
    if mode not in SUPPORTED_TIMER_MODES:
        raise ValueError(
            f"Unknown timer_data_mode: {raw_mode}. "
            f"Expected one of {sorted(SUPPORTED_TIMER_MODES)}"
        )
    return mode


def reshape_timer_dataframe(
    df: pd.DataFrame,
    channel_independence: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, int, int]:
    wide_df = ensure_datetime_index(df)
    if wide_df.shape[1] <= 0:
        raise ValueError("Input dataframe must contain at least one value column.")

    if channel_independence:
        series_num = wide_df.shape[1]
        series_dim = 1
        data_np = rearrange(wide_df.values, "t n -> t 1 n")
        sample_timestamp = wide_df.iloc[:, :1].copy()
        return data_np, sample_timestamp, series_num, series_dim

    series_num = 1
    series_dim = wide_df.shape[1]
    data_np = rearrange(wide_df.values, "t c -> t c 1")
    sample_timestamp = wide_df.iloc[:, :series_dim].copy()
    return data_np, sample_timestamp, series_num, series_dim


def fit_timer_scaler(
    train_data: np.ndarray,
    valid_data: Optional[np.ndarray],
    norm: bool,
) -> Tuple[StandardScaler, np.ndarray, Optional[np.ndarray]]:
    scaler = StandardScaler()
    train_len, channel_dim, series_num = train_data.shape
    scaler.fit(rearrange(train_data, "l c n -> (l n) c"))

    if not norm:
        return scaler, train_data, valid_data

    train_scaled = scaler.transform(rearrange(train_data, "l c n -> (l n) c"))
    train_data = rearrange(train_scaled, "(l n) c -> l c n", l=train_len, n=series_num)

    if valid_data is not None:
        valid_len = valid_data.shape[0]
        valid_scaled = scaler.transform(rearrange(valid_data, "l c n -> (l n) c"))
        valid_data = rearrange(valid_scaled, "(l n) c -> l c n", l=valid_len, n=series_num)

    return scaler, train_data, valid_data


def fit_timer_wide_scaler(
    train_data: pd.DataFrame,
    valid_data: Optional[pd.DataFrame],
    norm: bool,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[StandardScaler, pd.DataFrame, Optional[pd.DataFrame]]:
    scaler = StandardScaler() if scaler is None else scaler
    scaler.fit(train_data.to_numpy())

    if not norm:
        return scaler, train_data, valid_data

    train_scaled = pd.DataFrame(
        scaler.transform(train_data.to_numpy()),
        index=train_data.index,
        columns=train_data.columns,
    )

    valid_scaled = None
    if valid_data is not None:
        valid_scaled = pd.DataFrame(
            scaler.transform(valid_data.to_numpy()),
            index=valid_data.index,
            columns=valid_data.columns,
        )

    return scaler, train_scaled, valid_scaled


def build_timer_time_marks(index: pd.Index, timeenc: int, freq: str) -> np.ndarray:
    time_stamp = np.array([pd.to_datetime(index).to_numpy()])
    return get_time_mark(time_stamp, timeenc, freq)[0]


class TimerGroupedCIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config, *, timeenc: int = 1):
        self.data = ensure_datetime_index(data)
        self.values = self.data.to_numpy(dtype=np.float32, copy=True)
        self.time_marks = build_timer_time_marks(self.data.index, timeenc, config.freq)
        self.input_len = int(config.seq_len)
        self.label_len = int(config.label_len)
        self.pred_len = int(config.pred_len)
        self.series_num = int(getattr(config, "series_num", 1))
        self.series_dim = int(getattr(config, "series_dim", self.values.shape[1] // max(self.series_num, 1)))

        if self.series_num <= 1:
            raise ValueError("TimerGroupedCIDataset requires series_num > 1.")

        _validate_wide_shape(self.values, self.series_num, self.series_dim)
        self.n_timepoint = max(len(self.values) - self.input_len - self.pred_len + 1, 0)

    def __len__(self) -> int:
        return self.n_timepoint * self.series_dim

    def __getitem__(self, index: int):
        if self.n_timepoint <= 0:
            raise IndexError("TimerGroupedCIDataset is empty; there are no valid windows to sample.")

        window_index = index // self.series_dim
        channel_index = index % self.series_dim
        s_begin = window_index
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = np.zeros((self.series_num, self.input_len, 1), dtype=np.float32)
        seq_y = np.zeros((self.series_num, self.label_len + self.pred_len, 1), dtype=np.float32)
        for series_idx in range(self.series_num):
            col = series_idx * self.series_dim + channel_index
            seq_x[series_idx, :, 0] = self.values[s_begin:s_end, col]
            seq_y[series_idx, :, 0] = self.values[r_begin:r_end, col]

        seq_x_mark = self.time_marks[s_begin:s_end]
        seq_y_mark = self.time_marks[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )


def timer_ci_grouped_collate(batch):
    seq_x = torch.stack([item[0] for item in batch], dim=0)
    seq_y = torch.stack([item[1] for item in batch], dim=0)
    seq_x_mark = torch.stack([item[2] for item in batch], dim=0)
    seq_y_mark = torch.stack([item[3] for item in batch], dim=0)

    batch_size, series_num = seq_x.shape[:2]
    seq_x = rearrange(seq_x, "b n t c -> (b n) t c")
    seq_y = rearrange(seq_y, "b n t c -> (b n) t c")
    seq_x_mark = rearrange(seq_x_mark.unsqueeze(1).expand(-1, series_num, -1, -1), "b n t d -> (b n) t d")
    seq_y_mark = rearrange(seq_y_mark.unsqueeze(1).expand(-1, series_num, -1, -1), "b n t d -> (b n) t d")
    return seq_x, seq_y, seq_x_mark, seq_y_mark


class ChannelGroupedBatchSampler:
    def __init__(
        self,
        *,
        num_windows: int,
        series_dim: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        window_indices: Optional[Sequence[int]] = None,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.num_windows = int(num_windows)
        self.series_dim = int(series_dim)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.window_indices = list(range(self.num_windows)) if window_indices is None else [int(i) for i in window_indices]
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)

        if self.num_windows < 0:
            raise ValueError("num_windows must be non-negative.")
        if self.series_dim <= 0:
            raise ValueError("series_dim must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

    def _ordered_windows(self) -> list[int]:
        windows = self.window_indices
        if self.num_replicas > 1:
            sampler = DistributedSampler(
                windows,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=0,
                drop_last=self.drop_last,
            )
            return [windows[idx] for idx in sampler]

        if self.shuffle and len(windows) > 1:
            order = torch.randperm(len(windows)).tolist()
            return [windows[idx] for idx in order]
        return list(windows)

    def __iter__(self):
        ordered_windows = self._ordered_windows()
        for channel_index in range(self.series_dim):
            for start in range(0, len(ordered_windows), self.batch_size):
                chunk = ordered_windows[start : start + self.batch_size]
                if len(chunk) < self.batch_size and self.drop_last:
                    continue
                yield [window_index * self.series_dim + channel_index for window_index in chunk]

    def __len__(self) -> int:
        if self.num_replicas > 1:
            sampler = DistributedSampler(
                self.window_indices,
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=0,
                drop_last=self.drop_last,
            )
            window_count = len(sampler)
        else:
            window_count = len(self.window_indices)

        if self.drop_last:
            batches_per_channel = window_count // self.batch_size
        else:
            batches_per_channel = math.ceil(window_count / self.batch_size) if window_count > 0 else 0
        return batches_per_channel * self.series_dim


class TimerCIDataset(Dataset):
    def __init__(self, data: pd.DataFrame, config, *, timeenc: int = 1):
        self.data = ensure_datetime_index(data)
        self.values = self.data.to_numpy(dtype=np.float32, copy=True)
        self.time_marks = build_timer_time_marks(self.data.index, timeenc, config.freq)
        self.input_len = int(config.seq_len)
        self.label_len = int(config.label_len)
        self.pred_len = int(config.pred_len)

        self.n_var = self.values.shape[1]
        self.n_timepoint = max(len(self.values) - self.input_len - self.pred_len + 1, 0)

    def __len__(self) -> int:
        return self.n_var * self.n_timepoint

    def __getitem__(self, index: int):
        if self.n_timepoint <= 0:
            raise IndexError("TimerCIDataset is empty; there are no valid windows to sample.")

        c_begin = index // self.n_timepoint
        s_begin = index % self.n_timepoint
        s_end = s_begin + self.input_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.values[s_begin:s_end, c_begin : c_begin + 1]
        seq_y = self.values[r_begin:r_end, c_begin : c_begin + 1]
        seq_x_mark = self.time_marks[s_begin:s_end]
        seq_y_mark = self.time_marks[r_begin:r_end]

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
            torch.tensor(seq_x_mark, dtype=torch.float32),
            torch.tensor(seq_y_mark, dtype=torch.float32),
        )


def build_timer_ci_loaders(
    train_data: pd.DataFrame,
    valid_data: Optional[pd.DataFrame],
    config,
) -> Tuple[TimerCIDataset, DataLoader, Optional[DataLoader]]:
    series_num = int(getattr(config, "series_num", 1))
    if series_num > 1:
        train_dataset = TimerGroupedCIDataset(train_data, config, timeenc=1)
        train_windows = len(train_dataset) // train_dataset.series_dim
        window_indices: Sequence[int] = list(range(train_windows))
        if str(getattr(config, "shot_mode", "full_shot")).lower() == "few_shot":
            selected = build_few_shot_subset(
                list(range(train_windows)),
                getattr(config, "sampling_rate", getattr(config, "few_shot_ratio", 0.1)),
                strategy=getattr(config, "sampling_strategy", "uniform"),
                seed=getattr(config, "seed", None),
            )
            window_indices = list(getattr(selected, "indices", selected))

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=ChannelGroupedBatchSampler(
                num_windows=train_windows,
                series_dim=train_dataset.series_dim,
                batch_size=int(config.batch_size),
                shuffle=True,
                drop_last=False,
                window_indices=window_indices,
            ),
            num_workers=config.num_workers,
            collate_fn=timer_ci_grouped_collate,
        )

        valid_loader = None
        if valid_data is not None and len(valid_data) >= (config.seq_len + config.pred_len):
            valid_dataset = TimerGroupedCIDataset(valid_data, config, timeenc=1)
            valid_windows = len(valid_dataset) // valid_dataset.series_dim
            valid_loader = DataLoader(
                valid_dataset,
                batch_sampler=ChannelGroupedBatchSampler(
                    num_windows=valid_windows,
                    series_dim=valid_dataset.series_dim,
                    batch_size=int(config.batch_size),
                    shuffle=False,
                    drop_last=False,
                ),
                num_workers=config.num_workers,
                collate_fn=timer_ci_grouped_collate,
            )

        return train_dataset, train_loader, valid_loader

    train_dataset = TimerCIDataset(train_data, config, timeenc=1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=False,
    )
    train_loader = apply_few_shot_loader(
        dataset=train_dataset,
        loader=train_loader,
        config=config,
        drop_last=False,
        collate_fn=train_loader.collate_fn,
    )

    valid_loader = None
    if valid_data is not None and len(valid_data) >= (config.seq_len + config.pred_len):
        valid_dataset = TimerCIDataset(valid_data, config, timeenc=1)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )

    return train_dataset, train_loader, valid_loader
