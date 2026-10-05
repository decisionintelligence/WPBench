from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset, Sampler

from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.tsfm.submodules.toto.dataset_utils import (
    CausalMaskedTimeseries,
)
from ts_benchmark.baselines.tsfm.submodules.toto.models.finetune_lightning_module import (
    TotoForFinetuning,
)
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import infer_series_number


@dataclass
class TotoOfficialTrainerResult:
    best_ckpt_path: Optional[str]
    best_val_loss: Optional[float]
    train_windows: int
    val_windows: int
    run_dir: str


def _ensure_variate_first(values: np.ndarray | torch.Tensor) -> torch.Tensor:
    tensor = values if torch.is_tensor(values) else torch.as_tensor(values)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor.transpose(1, 0)
    raise ValueError(f"Expected 1D/2D time series, got shape={tuple(tensor.shape)}.")


def _preprocess_exogenous_features(
    past_exog: np.ndarray | torch.Tensor,
    future_exog: np.ndarray | torch.Tensor,
    *,
    context_length: int,
    patch_size: int,
) -> torch.Tensor:
    past_vt = _ensure_variate_first(past_exog)
    future_vt = _ensure_variate_first(future_exog)
    exogenous = torch.cat([past_vt, future_vt], dim=-1)
    exogenous = exogenous[..., -context_length:]
    padding = torch.zeros(
        (exogenous.shape[0], patch_size),
        dtype=exogenous.dtype,
        device=exogenous.device,
    )
    return torch.cat([exogenous, padding], dim=-1)


def collate_causal(batch: list[CausalMaskedTimeseries]) -> CausalMaskedTimeseries:
    series = torch.stack([b.series for b in batch], dim=0)
    padding_mask = torch.stack([b.padding_mask for b in batch], dim=0)
    id_mask = torch.stack([b.id_mask for b in batch], dim=0)
    timestamp_seconds = torch.stack([b.timestamp_seconds for b in batch], dim=0)
    time_interval_seconds = torch.stack([b.time_interval_seconds for b in batch], dim=0)
    return CausalMaskedTimeseries(
        series=series,
        padding_mask=padding_mask,
        id_mask=id_mask,
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
        input_slice=batch[0].input_slice,
        target_slice=batch[0].target_slice,
        num_exogenous_variables=batch[0].num_exogenous_variables,
    )


class TotoOfficialWindowDataset(Dataset):
    def __init__(
        self,
        *,
        target_data: np.ndarray,
        exogenous_data: Optional[np.ndarray],
        context_length: int,
        patch_size: int,
        window_indices: Optional[list[int]] = None,
    ):
        target_np = np.asarray(target_data, dtype=np.float32)
        if target_np.ndim != 3:
            raise ValueError(f"target_data must have shape (T, C, N), got {target_np.shape}")
        if context_length <= 0 or patch_size <= 0:
            raise ValueError("context_length and patch_size must be positive.")
        if target_np.shape[0] < context_length + patch_size:
            raise ValueError(
                "target_data is too short for the requested context_length + patch_size: "
                f"{target_np.shape[0]} < {context_length + patch_size}"
            )

        self.target_data = target_np
        self.exogenous_data = None if exogenous_data is None else np.asarray(exogenous_data, dtype=np.float32)
        if self.exogenous_data is not None and self.exogenous_data.shape[0] != self.target_data.shape[0]:
            raise ValueError("exogenous_data must align with target_data on the time axis.")
        if self.exogenous_data is not None and self.exogenous_data.shape[2] != self.target_data.shape[2]:
            raise ValueError("exogenous_data must align with target_data on the series axis.")

        self.context_length = int(context_length)
        self.patch_size = int(patch_size)
        self.series_num = int(target_np.shape[2])
        self.series_dim = int(target_np.shape[1])
        self.exogenous_dim = 0 if self.exogenous_data is None else int(self.exogenous_data.shape[1])
        self.total_windows = int(target_np.shape[0] - self.context_length - self.patch_size + 1)
        self.window_indices = (
            [int(idx) for idx in window_indices]
            if window_indices is not None
            else list(range(self.total_windows))
        )
        self.window_count = len(self.window_indices)
        self.group_size = self.series_num

    def __len__(self) -> int:
        return self.window_count * self.series_num

    def __getitem__(self, index: int) -> CausalMaskedTimeseries:
        local_window_index = int(index) // self.series_num
        series_index = int(index) % self.series_num
        window_index = self.window_indices[local_window_index]

        context_stop = window_index + self.context_length
        future_stop = context_stop + self.patch_size

        past_target = self.target_data[window_index:context_stop, :, series_index]
        future_target = self.target_data[context_stop:future_stop, :, series_index]
        past_target_vt = _ensure_variate_first(past_target)
        future_target_vt = _ensure_variate_first(future_target)
        series = torch.cat([past_target_vt, future_target_vt], dim=-1)
        padding_mask = torch.ones_like(series, dtype=torch.bool)
        num_exogenous_variables = 0

        if self.exogenous_data is not None:
            past_exog = self.exogenous_data[window_index:context_stop, :, series_index]
            future_exog = self.exogenous_data[context_stop:future_stop, :, series_index]
            exogenous = _preprocess_exogenous_features(
                past_exog,
                future_exog,
                context_length=self.context_length,
                patch_size=self.patch_size,
            )
            series = torch.cat([series, exogenous], dim=0)
            exog_padding = torch.ones_like(exogenous, dtype=torch.bool)
            exog_padding[:, -self.patch_size :] = False
            padding_mask = torch.cat([padding_mask, exog_padding], dim=0)
            num_exogenous_variables = self.exogenous_dim

        num_variates = int(series.shape[0])
        time_steps = int(series.shape[1])
        id_mask = torch.zeros((num_variates, time_steps), dtype=torch.long)
        timestamp_seconds = torch.arange(time_steps, dtype=torch.long).unsqueeze(0).expand(num_variates, time_steps)
        time_interval_seconds = torch.ones((num_variates,), dtype=torch.long)
        return CausalMaskedTimeseries(
            series=series.to(torch.float32),
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            input_slice=slice(0, self.context_length),
            target_slice=slice(self.patch_size, self.context_length + self.patch_size),
            num_exogenous_variables=num_exogenous_variables,
        )


class TotoOfficialWindowSampler(Sampler[int]):
    def __init__(
        self,
        *,
        window_count: int,
        group_size: int,
        windows_per_epoch: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.window_count = int(window_count)
        self.group_size = int(group_size)
        self.shuffle = bool(shuffle)
        self.windows_per_epoch = (
            int(windows_per_epoch)
            if windows_per_epoch is not None
            else int(window_count)
        )
        if self.window_count < 0:
            raise ValueError("window_count must be non-negative.")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive.")

    def __iter__(self):
        windows = list(range(self.window_count))
        if self.shuffle and len(windows) > 1:
            order = torch.randperm(len(windows)).tolist()
            windows = [windows[idx] for idx in order]
        windows = windows[: min(len(windows), self.windows_per_epoch)]
        for window_index in windows:
            start = int(window_index) * self.group_size
            yield from range(start, start + self.group_size)

    def __len__(self) -> int:
        return min(self.window_count, self.windows_per_epoch) * self.group_size


class _TotoOfficialDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: TotoOfficialWindowDataset,
        val_dataset: Optional[TotoOfficialWindowDataset],
        *,
        train_batch_windows: int,
        val_batch_windows: int,
        train_windows_per_epoch: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_windows = max(int(train_batch_windows), 1)
        self.val_batch_windows = max(int(val_batch_windows), 1)
        self.train_windows_per_epoch = max(int(train_windows_per_epoch), 1)
        self.num_workers = max(int(num_workers), 0)

    def train_dataloader(self) -> DataLoader:
        sampler = TotoOfficialWindowSampler(
            window_count=self.train_dataset.window_count,
            group_size=self.train_dataset.group_size,
            windows_per_epoch=self.train_windows_per_epoch,
            shuffle=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_windows * self.train_dataset.group_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_causal,
            pin_memory=True,
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_windows * self.val_dataset.group_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate_causal,
            pin_memory=True,
        )


def _resolve_official_value(config, official_attr: str, fallback_attr: str, default):
    value = getattr(config, official_attr, None)
    if value not in {None, "", "None"}:
        return value
    return getattr(config, fallback_attr, default)


def resolve_toto_trainer_device(config) -> str:
    return str(getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu"))


def resolve_toto_trainer_train_batch_count(
    *,
    windows_per_epoch: int,
    train_batch_windows: int,
) -> int:
    return max(1, math.ceil(max(int(windows_per_epoch), 1) / max(int(train_batch_windows), 1)))


def resolve_toto_trainer_val_check_interval(
    config,
    *,
    train_batch_count: int,
    requested: Optional[int] = None,
) -> int:
    value = requested
    if value in {None, "", "None"}:
        value = getattr(config, "toto_official_val_check_interval", None)
    if value in {None, "", "None"}:
        value = train_batch_count
    return max(1, min(int(value), int(train_batch_count)))


def _resolve_context_length(config, patch_size: int) -> int:
    requested = int(getattr(config, "seq_len", patch_size))
    context_length = (requested // patch_size) * patch_size
    if context_length <= 0:
        context_length = patch_size
    return int(context_length)


def _resolve_aligned_exog_wide(
    target_wide_df: pd.DataFrame,
    covariates,
    *,
    series_num: int,
) -> Optional[pd.DataFrame]:
    exog_df = (covariates or {}).get("exog", None)
    if exog_df is None:
        return None
    if not isinstance(exog_df, pd.DataFrame):
        raise ValueError("Covariates 'exog' must be a pandas DataFrame.")
    exog_df = exog_df.copy()
    if not exog_df.index.equals(target_wide_df.index):
        exog_df = exog_df.reindex(target_wide_df.index)
    if exog_df.isna().any(axis=None):
        raise ValueError("Exogenous covariates must align with target_wide_df index.")
    if exog_df.shape[1] % series_num != 0:
        raise ValueError("Exogenous columns cannot be evenly divided by series_num.")
    return exog_df


def _prepare_adapter_state(
    adapter,
    target_wide_df: pd.DataFrame,
    exog_wide_df: Optional[pd.DataFrame],
    *,
    series_num: int,
    adj_mx,
    **kwargs,
) -> tuple[np.ndarray, Optional[np.ndarray], int, int]:
    series_dim = target_wide_df.shape[1] // series_num
    if series_num * series_dim != target_wide_df.shape[1]:
        raise ValueError("Target columns cannot be evenly divided by series_num.")

    sample_train_valid_data = target_wide_df.iloc[:, :series_dim]
    exog_dim = 0
    if exog_wide_df is not None:
        exog_dim = exog_wide_df.shape[1] // series_num
        sample_exog_data = exog_wide_df.iloc[:, :exog_dim]
        sample_train_valid_data = pd.concat([sample_train_valid_data, sample_exog_data], axis=1)

    if sample_train_valid_data.shape[1] == 1:
        adapter.single_forecasting_hyper_param_tune(sample_train_valid_data)
    else:
        adapter.multi_forecasting_hyper_param_tune(sample_train_valid_data)

    adapter.config.num_nodes = series_num
    adapter.config.series_num = series_num
    adapter.config.series_dim = series_dim
    adapter.config.input_dim = series_dim + exog_dim
    adapter.config.output_dim = series_dim
    adapter.config.adj_mx = adj_mx
    adapter.config.geo_data = kwargs.get("geo_data", None)
    adapter.config.seq_len = _resolve_context_length(adapter.config, int(getattr(adapter.config, "patch_size", 64)))

    target_np = adapter.reshape_spatiotemporal(target_wide_df, series_num)[0].astype(np.float32)
    exog_np = None
    if exog_wide_df is not None:
        exog_np = adapter.reshape_spatiotemporal(exog_wide_df, series_num)[0].astype(np.float32)
    return target_np, exog_np, series_dim, exog_dim


def _fit_benchmark_scalers(
    adapter,
    *,
    train_target_np: np.ndarray,
    train_exog_np: Optional[np.ndarray],
) -> None:
    adapter.scaler1.fit(rearrange(train_target_np, "l c n -> (l n) c"))
    if train_exog_np is not None:
        adapter.scaler2.fit(rearrange(train_exog_np, "l c n -> (l n) c"))


def _transform_with_benchmark_scalers(
    adapter,
    *,
    target_np: np.ndarray,
    exog_np: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    target_len = int(target_np.shape[0])
    scaled_target = adapter.scaler1.transform(
        rearrange(target_np, "l c n -> (l n) c")
    )
    target_np = rearrange(scaled_target, "(l n) c -> l c n", l=target_len).astype(np.float32)

    if exog_np is None:
        return target_np, None

    exog_len = int(exog_np.shape[0])
    scaled_exog = adapter.scaler2.transform(
        rearrange(exog_np, "l c n -> (l n) c")
    )
    exog_np = rearrange(scaled_exog, "(l n) c -> l c n", l=exog_len).astype(np.float32)
    return target_np, exog_np


def prepare_toto_official_datasets(
    adapter,
    train_valid_data: pd.DataFrame,
    *,
    covariates=None,
    train_ratio_in_tv: float = 1.0,
    adj_mx=None,
    **kwargs,
) -> tuple[TotoOfficialWindowDataset, Optional[TotoOfficialWindowDataset], int]:
    target_wide_df = train_valid_data.copy()
    series_num = infer_series_number(target_wide_df)
    exog_wide_df = _resolve_aligned_exog_wide(
        target_wide_df,
        covariates,
        series_num=series_num,
    )
    target_np, exog_np, _, _ = _prepare_adapter_state(
        adapter,
        target_wide_df,
        exog_wide_df,
        series_num=series_num,
        adj_mx=adj_mx,
        **kwargs,
    )

    context_length = int(adapter.config.seq_len)
    patch_size = int(getattr(adapter.config, "patch_size", 64))
    train_target_np, valid_target_np = train_val_split(
        target_np,
        train_ratio_in_tv,
        context_length,
    )
    train_exog_np, valid_exog_np = (None, None)
    if exog_np is not None:
        train_exog_np, valid_exog_np = train_val_split(
            exog_np,
            train_ratio_in_tv,
            context_length,
        )

    if bool(getattr(adapter.config, "norm", False)):
        _fit_benchmark_scalers(
            adapter,
            train_target_np=train_target_np,
            train_exog_np=train_exog_np,
        )
        train_target_np, train_exog_np = _transform_with_benchmark_scalers(
            adapter,
            target_np=train_target_np,
            exog_np=train_exog_np,
        )
        if valid_target_np is not None:
            valid_target_np, valid_exog_np = _transform_with_benchmark_scalers(
                adapter,
                target_np=valid_target_np,
                exog_np=valid_exog_np,
            )

    train_window_indices = list(
        range(max(int(train_target_np.shape[0] - context_length - patch_size + 1), 0))
    )
    if not train_window_indices:
        raise ValueError(
            "Training split is too short for Toto official_trainer: "
            f"len={train_target_np.shape[0]}, context_length={context_length}, patch_size={patch_size}"
        )

    if str(getattr(adapter.config, "shot_mode", "full_shot")).lower() == "few_shot":
        subset = build_few_shot_subset(
            train_window_indices,
            getattr(adapter.config, "sampling_rate", getattr(adapter.config, "few_shot_ratio", 0.1)),
            strategy=getattr(adapter.config, "sampling_strategy", "uniform"),
            seed=getattr(adapter.config, "seed", None),
        )
        if hasattr(subset, "indices"):
            train_window_indices = [int(i) for i in subset.indices]
        else:
            train_window_indices = [int(i) for i in subset]

    train_dataset = TotoOfficialWindowDataset(
        target_data=train_target_np,
        exogenous_data=train_exog_np,
        context_length=context_length,
        patch_size=patch_size,
        window_indices=train_window_indices,
    )

    valid_dataset = None
    if valid_target_np is not None:
        valid_window_count = max(int(valid_target_np.shape[0] - context_length - patch_size + 1), 0)
        if valid_window_count > 0:
            valid_dataset = TotoOfficialWindowDataset(
                target_data=valid_target_np,
                exogenous_data=valid_exog_np,
                context_length=context_length,
                patch_size=patch_size,
                window_indices=[valid_window_count - 1],
            )

    windows_per_epoch = int(
        _resolve_official_value(
            adapter.config,
            "toto_official_num_train_samples",
            "toto_num_train_samples",
            getattr(adapter.config, "num_train_samples", 100),
        )
    )
    if windows_per_epoch <= 0:
        windows_per_epoch = train_dataset.window_count
    windows_per_epoch = min(train_dataset.window_count, windows_per_epoch)
    return train_dataset, valid_dataset, windows_per_epoch


def run_toto_official_trainer(
    adapter,
    train_valid_data: pd.DataFrame,
    *,
    covariates=None,
    train_ratio_in_tv: float = 1.0,
    adj_mx=None,
    **kwargs,
) -> TotoOfficialTrainerResult:
    if covariates is None:
        covariates = {}

    config = adapter.config
    seed = int(getattr(config, "seed", 42))
    seed_everything(seed, workers=True)

    train_dataset, valid_dataset, windows_per_epoch = prepare_toto_official_datasets(
        adapter,
        train_valid_data,
        covariates=covariates,
        train_ratio_in_tv=train_ratio_in_tv,
        adj_mx=adj_mx,
        **kwargs,
    )

    train_batch_windows = int(
        _resolve_official_value(config, "toto_official_train_batch_size", "batch_size", 4)
    )
    val_batch_windows = int(
        _resolve_official_value(config, "toto_official_val_batch_size", "batch_size", 1)
    )
    num_workers = int(
        _resolve_official_value(config, "toto_official_num_workers", "num_workers", 0)
    )

    datamodule = _TotoOfficialDataModule(
        train_dataset,
        valid_dataset,
        train_batch_windows=train_batch_windows,
        val_batch_windows=val_batch_windows,
        train_windows_per_epoch=windows_per_epoch,
        num_workers=num_workers,
    )

    run_dir = Path(tempfile.mkdtemp(prefix="toto_official_trainer_"))
    lightning_module = TotoForFinetuning(
        pretrained_backbone=adapter.model.model,
        val_prediction_len=int(
            _resolve_official_value(config, "toto_val_prediction_len", "horizon", 96)
        ),
        stable_steps=int(
            _resolve_official_value(config, "toto_official_stable_steps", "stable_steps", 1000)
        ),
        decay_steps=int(
            _resolve_official_value(config, "toto_official_decay_steps", "decay_steps", 1000)
        ),
        warmup_steps=int(
            _resolve_official_value(config, "toto_official_warmup_steps", "warmup_steps", 200)
        ),
        lr=float(_resolve_official_value(config, "toto_official_lr", "lr", 1e-4)),
        min_lr=float(_resolve_official_value(config, "toto_official_min_lr", "min_lr", 1e-5)),
        add_exogenous_features=train_dataset.exogenous_dim > 0,
    )
    lightning_module.to(resolve_toto_trainer_device(config))

    callbacks = [TQDMProgressBar(refresh_rate=int(getattr(config, "toto_official_refresh_rate", 1)))]
    checkpoint_dir = run_dir / "checkpoints"
    if valid_dataset is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch}-{step}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_on_train_epoch_end=False,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="{epoch}-{step}",
            save_top_k=1,
            save_last=True,
        )
    callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(save_dir=str(run_dir), name="logs")
    train_steps_per_epoch = resolve_toto_trainer_train_batch_count(
        windows_per_epoch=windows_per_epoch,
        train_batch_windows=train_batch_windows,
    )
    default_max_steps = train_steps_per_epoch * max(int(getattr(config, "num_epochs", 1)), 1)
    max_steps = int(
        _resolve_official_value(config, "toto_official_max_steps", "max_steps", default_max_steps)
    )
    val_check_interval = resolve_toto_trainer_val_check_interval(
        config,
        train_batch_count=train_steps_per_epoch,
    )

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(run_dir),
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "logger": logger,
        "callbacks": callbacks,
        "max_steps": max_steps,
        "log_every_n_steps": int(getattr(config, "toto_official_log_every_n_steps", 1)),
        "num_sanity_val_steps": int(getattr(config, "toto_official_num_sanity_val_steps", 0)),
        "enable_progress_bar": bool(getattr(config, "toto_official_enable_progress_bar", True)),
    }
    if valid_dataset is not None:
        trainer_kwargs["val_check_interval"] = int(val_check_interval)

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, datamodule=datamodule)

    best_ckpt_path = checkpoint_callback.best_model_path or checkpoint_callback.last_model_path or None
    best_val_loss = (
        float(checkpoint_callback.best_model_score.item())
        if checkpoint_callback.best_model_score is not None
        else None
    )
    if best_ckpt_path is not None:
        reloaded = TotoForFinetuning.load_from_checkpoint(
            checkpoint_path=best_ckpt_path,
            pretrained_backbone=adapter.model.model,
            map_location="cpu",
        )
        adapter.model.model = reloaded.model
    else:
        adapter.model.model = lightning_module.model

    return TotoOfficialTrainerResult(
        best_ckpt_path=best_ckpt_path,
        best_val_loss=best_val_loss,
        train_windows=train_dataset.window_count,
        val_windows=0 if valid_dataset is None else valid_dataset.window_count,
        run_dir=str(run_dir),
    )
