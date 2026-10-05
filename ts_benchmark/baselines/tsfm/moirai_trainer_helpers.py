from __future__ import annotations

import json
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DistributedSampler

from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.tsfm.timer_helpers import ensure_datetime_index, infer_timer_freq
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import infer_series_number

_SUBMODULE_ROOT = Path(__file__).resolve().parent / "submodules"
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

from uni2ts.data.builder.simple import (  # noqa: E402
    SimpleEvalDatasetBuilder,
    SimpleFinetuneDatasetBuilder,
    generate_eval_builder,
    generate_finetune_builder,
)
from uni2ts.data.loader import DataLoader as Uni2TSDataLoader  # noqa: E402
from uni2ts.loss.packed import PackedMSELoss, PackedNLLLoss, PackedNRMSELoss  # noqa: E402
from uni2ts.model.moirai.finetune import MoiraiFinetune  # noqa: E402


@dataclass
class OfficialTrainerResult:
    best_ckpt_path: Optional[str]
    best_val_packed_nll: Optional[float]
    train_windows: int
    val_windows: int
    debug_dir: Optional[str]
    run_dir: str


class _OfficialFineTuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        *,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        train_shuffle: bool,
        val_shuffle: bool,
        train_drop_last: bool,
        val_drop_last: bool,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = int(train_batch_size)
        self.val_batch_size = int(val_batch_size)
        self.num_workers = int(num_workers)
        self.train_shuffle = bool(train_shuffle)
        self.val_shuffle = bool(val_shuffle)
        self.train_drop_last = bool(train_drop_last)
        self.val_drop_last = bool(val_drop_last)

    def _per_device_batch_size(self, batch_size: int) -> int:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return max(int(batch_size), 1)
        world_size = max(int(getattr(trainer, "world_size", 1)), 1)
        accumulate = max(int(getattr(trainer, "accumulate_grad_batches", 1)), 1)
        return max(int(batch_size) // (world_size * accumulate), 1)

    def _build_loader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
    ) -> Uni2TSDataLoader:
        trainer = getattr(self, "trainer", None)
        world_size = int(getattr(trainer, "world_size", 1)) if trainer is not None else 1
        group_size = int(getattr(dataset, "group_size", 1))
        if group_size > 1:
            window_count = int(getattr(dataset, "window_count", len(dataset) // group_size))
            sampler = _WindowGroupedSampler(
                window_count=window_count,
                group_size=group_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_replicas=world_size,
                rank=int(getattr(trainer, "global_rank", 0)) if trainer is not None else 0,
            )
            return Uni2TSDataLoader(
                dataset=dataset,
                batch_size=self._per_device_batch_size(batch_size),
                batch_size_factor=group_size,
                cycle=False,
                num_batches_per_epoch=None,
                shuffle=False,
                sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=None,
                pin_memory=pin_memory,
                drop_last=drop_last,
                fill_last=False,
                worker_init_fn=None,
                prefetch_factor=2,
                persistent_workers=self.num_workers > 0,
            )

        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=None,
                rank=None,
                shuffle=shuffle,
                seed=0,
                drop_last=drop_last,
            )
            if world_size > 1
            else None
        )
        return Uni2TSDataLoader(
            dataset=dataset,
            batch_size=self._per_device_batch_size(batch_size),
            batch_size_factor=2.0,
            cycle=False,
            num_batches_per_epoch=None,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=None,
            pin_memory=pin_memory,
            drop_last=drop_last,
            fill_last=False,
            worker_init_fn=None,
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> Uni2TSDataLoader:
        return self._build_loader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.train_shuffle,
            pin_memory=True,
            drop_last=self.train_drop_last,
        )

    def val_dataloader(self) -> Optional[Uni2TSDataLoader]:
        if self.val_dataset is None:
            return None
        return self._build_loader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            pin_memory=False,
            drop_last=self.val_drop_last,
        )


class _SeriesUnfoldDataset(Dataset):
    """
    Expand each base sample into per-series samples using series-major variate blocks.
    Expected ordering: [s0_c0..s0_c{C-1}, s1_c0.., ...].
    """

    def __init__(self, base_dataset: Dataset, *, series_num: int, series_dim: int):
        self.base_dataset = base_dataset
        self.series_num = int(series_num)
        self.series_dim = int(series_dim)
        if self.series_num <= 1:
            raise ValueError("_SeriesUnfoldDataset requires series_num > 1.")
        if self.series_dim <= 0:
            raise ValueError("_SeriesUnfoldDataset requires series_dim > 0.")
        self.group_size = self.series_num
        self.window_count = len(self.base_dataset)

    def __len__(self) -> int:
        return len(self.base_dataset) * self.series_num

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = idx // self.series_num
        series_idx = idx % self.series_num
        sample = self.base_dataset[base_idx]
        if "variate_id" not in sample:
            raise ValueError("Expected 'variate_id' in sample for series unfolding.")

        variate = sample["variate_id"]
        variate_tensor = (
            variate.detach().cpu()
            if torch.is_tensor(variate)
            else torch.as_tensor(np.asarray(variate))
        )
        if variate_tensor.ndim != 1:
            raise ValueError(
                f"Expected 1D variate_id for unfolding, got shape={tuple(variate_tensor.shape)}."
            )

        start = int(series_idx * self.series_dim)
        end = int(start + self.series_dim)
        mask = (variate_tensor >= start) & (variate_tensor < end)
        if not torch.any(mask):
            raise ValueError(
                "No variates selected for requested series block. "
                f"series_idx={series_idx}, block=[{start},{end}), "
                f"variate_min={int(variate_tensor.min().item())}, "
                f"variate_max={int(variate_tensor.max().item())}."
            )

        seq_len = int(variate_tensor.shape[0])
        out = {}
        for key, value in sample.items():
            if torch.is_tensor(value):
                if value.ndim >= 1 and int(value.shape[0]) == seq_len:
                    out[key] = value[mask.to(value.device)]
                else:
                    out[key] = value
            elif isinstance(value, np.ndarray):
                if value.ndim >= 1 and int(value.shape[0]) == seq_len:
                    out[key] = value[mask.numpy()]
                else:
                    out[key] = value
            else:
                out[key] = value

        if torch.is_tensor(out["variate_id"]):
            out["variate_id"] = out["variate_id"] - start
        else:
            out["variate_id"] = np.asarray(out["variate_id"]) - start
        return out


class _WindowGroupedSampler:
    def __init__(
        self,
        *,
        window_count: int,
        group_size: int,
        shuffle: bool,
        drop_last: bool,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.window_count = int(window_count)
        self.group_size = int(group_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.num_replicas = max(int(num_replicas), 1)
        self.rank = int(rank)

        if self.window_count < 0:
            raise ValueError("window_count must be non-negative.")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive.")

    def _ordered_windows(self) -> list[int]:
        windows = list(range(self.window_count))
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
        return windows

    def __iter__(self):
        for window_index in self._ordered_windows():
            start = window_index * self.group_size
            yield from range(start, start + self.group_size)

    def __len__(self) -> int:
        if self.num_replicas > 1:
            sampler = DistributedSampler(
                list(range(self.window_count)),
                num_replicas=self.num_replicas,
                rank=self.rank,
                shuffle=self.shuffle,
                seed=0,
                drop_last=self.drop_last,
            )
            window_count = len(sampler)
        else:
            window_count = self.window_count
        return window_count * self.group_size


def _resolve_mode(config, train_valid_data: pd.DataFrame) -> str:
    target_dim = int(getattr(config, "target_dim", max(1, train_valid_data.shape[1])))
    return "M" if target_dim > 1 or train_valid_data.shape[1] > 1 else "S"


def _resolve_official_value(config, official_attr: str, fallback_attr: str, default):
    value = getattr(config, official_attr, None)
    if value not in {None, "", "None"}:
        return value
    return getattr(config, fallback_attr, default)


def _tensor_summary(value: torch.Tensor) -> dict:
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() == 0:
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "numel": 0,
            "mean": None,
            "std": None,
            "head": [],
        }

    cast = flat.to(torch.float32)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(flat.numel()),
        "mean": float(cast.mean().item()),
        "std": float(cast.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
        "head": cast[: min(8, flat.numel())].tolist(),
    }


def _dump_first_batch(datamodule: _OfficialFineTuneDataModule, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _write(split: str, loader) -> None:
        if loader is None:
            return
        batch = next(iter(loader))
        summary = {
            key: _tensor_summary(value)
            for key, value in batch.items()
            if torch.is_tensor(value)
        }
        (output_dir / f"{split}_first_batch.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    _write("train", datamodule.train_dataloader())
    _write("val", datamodule.val_dataloader())


def _load_best_checkpoint(module: L.LightningModule, ckpt_path: Optional[str]) -> None:
    if not ckpt_path:
        return
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    module.load_state_dict(state_dict, strict=True)


def _validate_inputs(train_valid_data: pd.DataFrame, covariates) -> None:
    non_null_covariates = {
        key: value for key, value in (covariates or {}).items() if value is not None
    }
    unsupported_keys = sorted(key for key in non_null_covariates if key != "exog")
    if unsupported_keys:
        raise ValueError(
            "Moirai official_trainer currently supports only exog covariates; "
            f"got unsupported covariates={unsupported_keys}."
        )

    if train_valid_data.shape[1] <= 0:
        raise ValueError("train_valid_data must contain at least one target column.")


def _append_exog_as_training_targets(
    train_valid_data: pd.DataFrame,
    exog_df: Optional[pd.DataFrame],
    *,
    series_num: int,
) -> pd.DataFrame:
    if exog_df is None:
        return train_valid_data

    wide_df = ensure_datetime_index(train_valid_data)
    if wide_df.shape[-1] % series_num != 0:
        raise ValueError("Target columns cannot be evenly divided by series_num.")

    target_dim = wide_df.shape[-1] // series_num
    exog_dim = exog_df.shape[-1] // series_num

    target_np = wide_df.to_numpy(copy=True).reshape(len(wide_df), series_num, target_dim)
    exog_np = exog_df.to_numpy(copy=True).reshape(len(exog_df), series_num, exog_dim)
    merged_np = torch.from_numpy(target_np)
    merged_exog = torch.from_numpy(exog_np)
    merged_np = torch.cat([merged_np, merged_exog], dim=2).numpy().reshape(len(wide_df), -1)

    target_cols = list(wide_df.columns)
    exog_cols = list(exog_df.columns)
    merged_cols = []
    for series_idx in range(series_num):
        tgt_start = series_idx * target_dim
        exog_start = series_idx * exog_dim
        merged_cols.extend(target_cols[tgt_start : tgt_start + target_dim])
        merged_cols.extend(exog_cols[exog_start : exog_start + exog_dim])

    return pd.DataFrame(merged_np, index=wide_df.index, columns=merged_cols)


def _resolve_aligned_exog_wide(
    train_valid_data: pd.DataFrame,
    covariates,
    *,
    series_num: int,
) -> Optional[pd.DataFrame]:
    exog_data = (covariates or {}).get("exog", None)
    if exog_data is None:
        return None

    if not isinstance(exog_data, pd.DataFrame):
        raise ValueError("Covariates 'exog' must be a pandas DataFrame for Moirai official_trainer.")

    wide_df = ensure_datetime_index(train_valid_data)
    exog_df = ensure_datetime_index(exog_data)

    if wide_df.shape[-1] % series_num != 0:
        raise ValueError("Target columns cannot be evenly divided by series_num.")
    if exog_df.shape[-1] % series_num != 0:
        raise ValueError("Exog columns cannot be evenly divided by series_num.")

    if not exog_df.index.equals(wide_df.index):
        exog_df = exog_df.reindex(wide_df.index)
    if exog_df.isna().any(axis=None):
        raise ValueError(
            "Exog index must align with train_valid_data index for official_trainer; "
            "reindexing introduced NaN values."
        )

    return exog_df


def _prepare_benchmark_state(
    adapter,
    train_valid_data: pd.DataFrame,
    *,
    target_wide_df: pd.DataFrame,
    exog_wide_df: Optional[pd.DataFrame],
    series_num: int,
    covariates,
    train_ratio_in_tv: float,
    adj_mx,
    **kwargs,
) -> None:
    adapter.config.num_nodes = series_num
    series_dim = target_wide_df.shape[-1] // series_num
    if series_num * series_dim != target_wide_df.shape[-1]:
        raise ValueError("Target columns cannot be evenly divided by series_num.")
    if series_num * (train_valid_data.shape[-1] // series_num) != train_valid_data.shape[-1]:
        raise ValueError("Data columns cannot be evenly divided by series_num.")

    sample_train_valid_data = target_wide_df.iloc[:, :series_dim]
    train_valid_np = adapter.reshape_spatiotemporal(train_valid_data, series_num)[0]

    del covariates
    exog_dim = 0
    if exog_wide_df is not None:
        exog_dim = exog_wide_df.shape[-1] // series_num
        if series_num * exog_dim != exog_wide_df.shape[-1]:
            raise ValueError("Exogenous columns cannot be evenly divided by series_num.")
        sample_exog_data = exog_wide_df.iloc[:, :exog_dim]
        sample_train_valid_data = pd.concat([sample_train_valid_data, sample_exog_data], axis=1)

    if sample_train_valid_data.shape[1] == 1:
        adapter.single_forecasting_hyper_param_tune(sample_train_valid_data)
    else:
        adapter.multi_forecasting_hyper_param_tune(sample_train_valid_data)

    adapter.config.series_dim = series_dim
    adapter.config.input_dim = series_dim + exog_dim
    adapter.config.output_dim = series_dim
    adapter.config.adj_mx = adj_mx
    adapter.config.series_num = series_num
    adapter.config.geo_data = kwargs.get("geo_data", None)

    train_data, _ = train_val_split(
        train_valid_np,
        train_ratio_in_tv,
        adapter.config.seq_len,
    )
    if exog_dim > 0:
        adapter.scaler1.fit(train_data[:, :series_dim, :].transpose(0, 2, 1).reshape(-1, series_dim))
        adapter.scaler2.fit(train_data[:, series_dim:, :].transpose(0, 2, 1).reshape(-1, exog_dim))
    else:
        adapter.scaler1.fit(train_data.transpose(0, 2, 1).reshape(-1, series_dim))


def _build_official_finetune_module(base_model, config) -> MoiraiFinetune:
    patch_size = base_model.forecast_model.hparams.patch_size
    if patch_size == "auto":
        raise ValueError("official_trainer requires a fixed integer patch_size, not 'auto'.")

    shared_module = base_model.forecast_model.module
    official_module = MoiraiFinetune(
        min_patches=int(getattr(config, "min_patches", 2)),
        min_mask_ratio=float(getattr(config, "min_mask_ratio", 0.15)),
        max_mask_ratio=float(getattr(config, "max_mask_ratio", 0.5)),
        max_dim=int(getattr(config, "max_dim", 128)),
        num_training_steps=getattr(config, "num_training_steps", None),
        num_warmup_steps=int(getattr(config, "num_warmup_steps", 0)),
        module=shared_module,
        num_samples=int(getattr(config, "num_samples", 100)),
        beta1=float(getattr(config, "beta1", 0.9)),
        beta2=float(getattr(config, "beta2", 0.98)),
        loss_func=PackedNLLLoss(),
        val_metric=[
            PackedMSELoss(),
            PackedNRMSELoss(normalize="absolute_target_squared"),
        ],
        lr=float(getattr(config, "lr", 5e-7)),
        weight_decay=float(getattr(config, "weight_decay", 1e-1)),
        log_on_step=False,
        context_length=int(getattr(config, "seq_len", base_model.context_length)),
        prediction_length=int(getattr(config, "horizon", base_model.pred_len)),
        patch_size=int(patch_size),
        finetune_pattern=str(getattr(config, "finetune_pattern", "full")).lower(),
    )
    base_model.finetune_model = official_module
    return official_module


def _prepare_official_datasets(
    official_module: MoiraiFinetune,
    train_valid_data: pd.DataFrame,
    config,
    *,
    train_length: int,
    eval_length: int,
    storage_path: Path,
) -> tuple[Dataset, Optional[Dataset]]:
    wide_df = ensure_datetime_index(train_valid_data)
    freq = infer_timer_freq(wide_df.index, fallback=str(getattr(config, "freq", "h")))
    freq = str(freq).upper()
    mode = _resolve_mode(config, wide_df)
    dataset_type = "wide_multivariate" if mode == "M" else "wide"

    dataset_name = f"moirai_align_{uuid.uuid4().hex[:8]}"
    csv_path = storage_path / f"{dataset_name}.csv"
    wide_df.to_csv(csv_path)

    raw_train_builder = SimpleFinetuneDatasetBuilder(
        dataset=dataset_name,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
        mode=mode,
        storage_path=storage_path,
    )
    raw_train_builder.build_dataset(
        file=csv_path,
        dataset_type=dataset_type,
        offset=train_length,
        freq=freq,
        normalize=bool(getattr(config, "norm", True)),
    )

    eval_dataset_name = f"{dataset_name}_eval"
    if eval_length > 0:
        raw_eval_builder = SimpleEvalDatasetBuilder(
            dataset=eval_dataset_name,
            offset=None,
            windows=None,
            distance=None,
            prediction_length=None,
            context_length=None,
            patch_size=None,
            mode=mode,
            storage_path=storage_path,
        )
        raw_eval_builder.build_dataset(
            file=csv_path,
            dataset_type=dataset_type,
            freq=freq,
            mean=raw_train_builder.mean,
            std=raw_train_builder.std,
        )

    train_builder = generate_finetune_builder(
        dataset=dataset_name,
        train_length=train_length,
        prediction_length=int(getattr(config, "horizon", official_module.prediction_length)),
        context_length=int(getattr(config, "seq_len", official_module.context_length)),
        patch_size=int(getattr(config, "patch_size", official_module.patch_size)),
        mode=mode,
        storage_path=storage_path,
        distance=1,
    )
    train_dataset = train_builder.load_dataset(official_module.train_transform_map)

    val_dataset = None
    if eval_length > 0:
        val_builder = generate_eval_builder(
            dataset=eval_dataset_name,
            offset=train_length,
            eval_length=eval_length,
            prediction_length=int(getattr(config, "horizon", official_module.prediction_length)),
            context_length=int(getattr(config, "seq_len", official_module.context_length)),
            patch_size=int(getattr(config, "patch_size", official_module.patch_size)),
            mode=mode,
            storage_path=storage_path,
            distance=1,
        )
        val_dataset = val_builder.load_dataset(official_module.val_transform_map)

    if str(getattr(config, "shot_mode", "full_shot")).lower() == "few_shot":
        train_dataset = build_few_shot_subset(
            train_dataset,
            getattr(config, "sampling_rate", getattr(config, "few_shot_ratio", 0.1)),
            strategy=getattr(config, "sampling_strategy", "uniform"),
            seed=getattr(config, "seed", None),
        )

    return train_dataset, val_dataset


def run_moirai_official_trainer(
    adapter,
    train_valid_data: pd.DataFrame,
    *,
    covariates=None,
    train_ratio_in_tv: float = 1.0,
    adj_mx=None,
    **kwargs,
) -> OfficialTrainerResult:
    wide_df = ensure_datetime_index(train_valid_data)
    series_num = infer_series_number(wide_df)
    exog_wide_df = _resolve_aligned_exog_wide(wide_df, covariates, series_num=series_num)
    train_wide_df = _append_exog_as_training_targets(
        wide_df,
        exog_wide_df,
        series_num=series_num,
    )
    _validate_inputs(train_wide_df, covariates)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    config = adapter.config
    seed = int(getattr(config, "seed", 0))
    L.seed_everything(seed, workers=True)

    _prepare_benchmark_state(
        adapter,
        train_wide_df,
        target_wide_df=wide_df,
        exog_wide_df=exog_wide_df,
        series_num=series_num,
        covariates=covariates,
        train_ratio_in_tv=train_ratio_in_tv,
        adj_mx=adj_mx,
        **kwargs,
    )

    total_length = len(train_wide_df)
    train_length = int(total_length * float(train_ratio_in_tv))
    eval_length = total_length - train_length
    if train_length <= int(getattr(config, "seq_len", 0)) + int(getattr(config, "horizon", 0)):
        raise ValueError(
            "official_trainer split is too small for the requested seq_len and horizon: "
            f"train_length={train_length}, seq_len={getattr(config, 'seq_len', None)}, horizon={getattr(config, 'horizon', None)}"
        )

    base_model = adapter.model.module if isinstance(adapter.model, torch.nn.DataParallel) else adapter.model
    official_module = _build_official_finetune_module(base_model, config)
    if "sample_id" not in official_module.seq_fields:
        official_module.seq_fields = official_module.seq_fields + ("sample_id",)

    tmp_root = Path(tempfile.mkdtemp(prefix="moirai_official_align_"))
    storage_path = tmp_root / "storage"
    storage_path.mkdir(parents=True, exist_ok=True)
    run_dir = tmp_root / "trainer"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset = _prepare_official_datasets(
        official_module,
        train_wide_df,
        config,
        train_length=train_length,
        eval_length=eval_length,
        storage_path=storage_path,
    )

    unfold_series_in_batch = bool(getattr(config, "official_unfold_series_in_batch", True))
    if unfold_series_in_batch and int(getattr(config, "series_num", 1)) > 1:
        per_series_dim = int(getattr(config, "series_dim", 0))
        train_dataset = _SeriesUnfoldDataset(
            train_dataset,
            series_num=int(getattr(config, "series_num", 1)),
            series_dim=per_series_dim,
        )
        if val_dataset is not None:
            val_dataset = _SeriesUnfoldDataset(
                val_dataset,
                series_num=int(getattr(config, "series_num", 1)),
                series_dim=per_series_dim,
            )

    train_windows = int(getattr(train_dataset, "window_count", len(train_dataset)))
    val_windows = int(getattr(val_dataset, "window_count", len(val_dataset))) if val_dataset is not None else 0

    batch_size = int(_resolve_official_value(config, "official_batch_size", "batch_size", 64))
    num_epochs = int(_resolve_official_value(config, "official_num_epochs", "num_epochs", 200))
    patience = int(_resolve_official_value(config, "official_patience", "patience", 3))
    gradient_clip_val = float(
        _resolve_official_value(config, "official_gradient_clip_val", "gradient_clip_val", 1.0)
    )
    num_workers = int(
        _resolve_official_value(config, "official_trainer_num_workers", "num_workers", 0)
    )

    grouped_series_batches = unfold_series_in_batch and int(getattr(config, "series_num", 1)) > 1
    train_drop_last = False if grouped_series_batches else bool(getattr(config, "series_dim", train_wide_df.shape[1]) != 1)
    val_drop_last = False
    train_shuffle = True
    val_shuffle = True

    datamodule = _OfficialFineTuneDataModule(
        train_dataset,
        val_dataset,
        train_batch_size=batch_size,
        val_batch_size=batch_size,
        num_workers=num_workers,
        train_shuffle=train_shuffle,
        val_shuffle=val_shuffle,
        train_drop_last=train_drop_last,
        val_drop_last=val_drop_last,
    )

    debug_dir = None
    if bool(getattr(config, "official_debug_dump_first_batch", False)):
        repo_root = Path(__file__).resolve().parents[3]
        debug_dir_path = (
            repo_root
            / "result"
            / "debug"
            / "moirai_official_align"
            / f"{uuid.uuid4().hex[:8]}_pl{getattr(config, 'horizon', 'na')}_cl{getattr(config, 'seq_len', 'na')}"
        )
        debug_dir = str(debug_dir_path)
        _dump_first_batch(datamodule, debug_dir_path)
        (debug_dir_path / "metadata.json").write_text(
            json.dumps(
                {
                    "train_length": train_length,
                    "eval_length": eval_length,
                    "train_windows": train_windows,
                    "val_windows": val_windows,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "patience": patience,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        # The debug dump iterates one training batch, so reseed afterwards to
        # keep the actual trainer run identical to the no-dump path.
        L.seed_everything(seed, workers=True)

    callbacks = [LearningRateMonitor(logging_interval="epoch")]
    if val_dataset is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            monitor="val/PackedNLLLoss",
            save_weights_only=True,
            mode="min",
            save_top_k=1,
            every_n_epochs=1,
            save_last=False,
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(
            EarlyStopping(
                monitor="val/PackedNLLLoss",
                min_delta=0.0,
                patience=patience,
                mode="min",
                strict=True,
                verbose=True,
            )
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            save_weights_only=True,
            save_top_k=1,
            every_n_epochs=1,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    logger = TensorBoardLogger(save_dir=str(run_dir), name="logs")
    trainer = L.Trainer(
        default_root_dir=str(run_dir),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        num_nodes=1,
        precision=32,
        logger=logger,
        callbacks=callbacks,
        max_epochs=num_epochs,
        enable_progress_bar=True,
        accumulate_grad_batches=1,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    print("Number of windows in finetune: ", train_windows)
    print("Batch size for finetune: ", batch_size)
    print("Number of batches in a epoch: ", max(train_windows // max(batch_size, 1), 0))
    if val_dataset is not None:
        print("Number of windows in val: ", val_windows)
        print("Batch size for val: ", batch_size)
        print("Number of batches in a epoch: ", max(val_windows // max(batch_size, 1), 0))

    trainer.fit(official_module, datamodule=datamodule)

    best_model_path = checkpoint_callback.best_model_path or getattr(checkpoint_callback, "last_model_path", "")
    best_score = checkpoint_callback.best_model_score
    best_val = float(best_score.item()) if best_score is not None else None
    if val_dataset is not None and not best_model_path:
        raise RuntimeError("official_trainer did not produce a best checkpoint for validation monitoring.")
    if val_dataset is not None and best_val is None:
        raise RuntimeError("official_trainer did not report best val/PackedNLLLoss.")
    _load_best_checkpoint(official_module, best_model_path)
    base_model.finetune_model = official_module

    return OfficialTrainerResult(
        best_ckpt_path=best_model_path or None,
        best_val_packed_nll=best_val,
        train_windows=train_windows,
        val_windows=val_windows,
        debug_dir=debug_dir,
        run_dir=str(run_dir),
    )
