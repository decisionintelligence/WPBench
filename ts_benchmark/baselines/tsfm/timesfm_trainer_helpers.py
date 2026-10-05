import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset


class TimesFMBenchmarkDataset(Dataset):
    def __init__(self, data_2d, seq_len, horizon, freq_type, patch_len=32, series_num=1, series_dim=None, window_indices=None):
        self.data_2d = np.asarray(data_2d, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.freq_type = int(freq_type)
        self.patch_len = int(patch_len)
        self.series_num = int(series_num)
        self.series_dim = int(series_dim) if series_dim is not None else self.data_2d.shape[1] // max(self.series_num, 1)
        self.seq_len_aligned = ((self.seq_len + self.patch_len - 1) // self.patch_len) * self.patch_len
        self.pad_left = self.seq_len_aligned - self.seq_len
        self.total_windows = max(self.data_2d.shape[0] - self.seq_len - self.horizon + 1, 0)

        if self.series_num == 1:
            self.samples = []
            for col in range(self.data_2d.shape[1]):
                series = self.data_2d[:, col].astype(np.float32)
                for start in range(0, len(series) - seq_len - horizon + 1):
                    ctx = series[start : start + seq_len]
                    fut = series[start + seq_len : start + seq_len + horizon]
                    padding = np.zeros(self.seq_len_aligned, dtype=np.float32)
                    if self.pad_left > 0:
                        ctx = np.pad(ctx, (self.pad_left, 0), constant_values=0.0)
                        padding[: self.pad_left] = 1.0
                    self.samples.append((ctx, padding, fut))
            return

        if self.series_num * self.series_dim != self.data_2d.shape[1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        self.window_indices = list(range(self.total_windows)) if window_indices is None else [int(i) for i in window_indices]
        self.window_count = len(self.window_indices)

    def _build_grouped_sample(self, window_index: int, channel_index: int):
        # One dataset item now represents one window for one channel, and keeps
        # all series aligned in the channel dimension so collate can expand it
        # to the expected batch_size x series_num layout.
        channel_columns = [series_idx * self.series_dim + channel_index for series_idx in range(self.series_num)]
        past = self.data_2d[window_index : window_index + self.seq_len, channel_columns]
        future = self.data_2d[
            window_index + self.seq_len : window_index + self.seq_len + self.horizon,
            channel_columns,
        ]

        padding = np.zeros((self.series_num, self.seq_len_aligned), dtype=np.float32)
        if self.pad_left > 0:
            past = np.pad(past, ((self.pad_left, 0), (0, 0)), constant_values=0.0)
            padding[:, : self.pad_left] = 1.0

        return rearrange(past, "t n -> n t"), padding, rearrange(future, "t n -> n t")

    def __len__(self):
        if self.series_num == 1:
            return len(self.samples)
        return self.window_count * self.series_dim

    def __getitem__(self, idx):
        if self.series_num == 1:
            x_context, x_padding, x_future = self.samples[idx]
        else:
            window_index = self.window_indices[idx // self.series_dim]
            channel_index = idx % self.series_dim
            x_context, x_padding, x_future = self._build_grouped_sample(window_index, channel_index)

        x_context = torch.tensor(x_context, dtype=torch.float32)
        x_padding = torch.tensor(x_padding, dtype=torch.float32)
        if self.series_num == 1:
            freq = torch.tensor([self.freq_type], dtype=torch.long)
        else:
            freq = torch.full((self.series_num, 1), self.freq_type, dtype=torch.long)
        x_future = torch.tensor(x_future, dtype=torch.float32)
        return x_context, x_padding, freq, x_future
