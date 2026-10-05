import inspect
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.utils import train_val_split
from ts_benchmark.utils.data_processing import infer_series_number


class TinyTimeMixerTrainerDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        pred_len: int,
        *,
        series_num: int = 1,
        use_frequency_token: bool = False,
        freq_token_id: int = 0,
    ):
        if data.ndim != 3:
            raise ValueError(f"Expected data with shape (T, C, N), got {data.shape}")
        if data.shape[0] < seq_len + pred_len:
            raise ValueError(
                f"Data length {data.shape[0]} is shorter than seq_len + pred_len = {seq_len + pred_len}."
            )

        self.data = data.astype(np.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.use_frequency_token = bool(use_frequency_token)
        self.freq_token_id = int(freq_token_id)
        self.series_num = int(series_num)
        self.total_windows = max(self.data.shape[0] - self.seq_len - self.pred_len + 1, 0)
        self.series_dim = self.data.shape[1]
        if self.series_num <= 0:
            raise ValueError("series_num must be positive.")
        if self.data.shape[2] != self.series_num:
            raise ValueError(
                f"Expected data with {self.series_num} series, got {self.data.shape[2]}."
            )

    def __len__(self) -> int:
        if self.series_num > 1:
            return self.total_windows
        return self.total_windows * self.series_num

    def __getitem__(self, index):
        if self.series_num > 1:
            window_index = index
            past = self.data[window_index : window_index + self.seq_len]
            future = self.data[
                window_index + self.seq_len : window_index + self.seq_len + self.pred_len
            ]
            item = {
                "past_values": torch.tensor(rearrange(past, "t c n -> n t c"), dtype=torch.float32),
                "future_values": torch.tensor(rearrange(future, "t c n -> n t c"), dtype=torch.float32),
            }
            if self.use_frequency_token:
                item["freq_token"] = torch.full((self.series_num,), self.freq_token_id, dtype=torch.long)
            return item

        window_index = index // self.series_num
        node_index = index % self.series_num
        past = self.data[window_index : window_index + self.seq_len, :, node_index]
        future = self.data[
            window_index + self.seq_len : window_index + self.seq_len + self.pred_len,
            :,
            node_index,
        ]

        item = {
            "past_values": torch.tensor(past, dtype=torch.float32),
            "future_values": torch.tensor(future, dtype=torch.float32),
        }
        if self.use_frequency_token:
            item["freq_token"] = torch.tensor(self.freq_token_id, dtype=torch.long)
        return item


def ttm_grouped_collate(batch):
    first_item = batch[0]
    grouped = torch.is_tensor(first_item["past_values"]) and first_item["past_values"].ndim == 3

    past_values = torch.stack([item["past_values"] for item in batch], dim=0)
    future_values = torch.stack([item["future_values"] for item in batch], dim=0)

    if grouped:
        batch_size, series_num, seq_len, channel_dim = past_values.shape
        past_values = past_values.reshape(batch_size * series_num, seq_len, channel_dim)
        future_values = future_values.reshape(batch_size * series_num, future_values.shape[2], future_values.shape[3])
    result = {
        "past_values": past_values,
        "future_values": future_values,
    }

    if "freq_token" in first_item:
        freq_token = torch.stack([item["freq_token"] for item in batch], dim=0)
        if grouped:
            freq_token = freq_token.reshape(-1)
        result["freq_token"] = freq_token

    return result


def prepare_ttm_trainer_datasets(
    adapter,
    train_valid_data,
    *,
    covariates=None,
    train_ratio_in_tv: float = 1.0,
    adj_mx=None,
    **kwargs,
) -> Tuple[Dataset, Optional[Dataset]]:
    if covariates is None:
        covariates = {}

    # Infer how many parallel series/nodes are packed into the flattened dataframe.
    series_num = infer_series_number(train_valid_data)
    adapter.config.num_nodes = series_num
    series_dim = train_valid_data.shape[-1] // series_num
    if series_num * series_dim != train_valid_data.shape[-1]:
        raise ValueError("Data columns cannot be evenly divided by series_num.")

    sample_train_valid_data = train_valid_data.iloc[:, :series_dim]

    # Reuse the benchmark reshape path: (T, N*C) -> (T, C, N).
    train_valid_np = adapter.reshape_spatiotemporal(train_valid_data, series_num)[0]

    # Match Timer's covariate handling: append exogenous channels to the input
    # representation, but keep output_dim anchored to the target series only.
    exog_data = covariates.get("exog", None)
    if exog_data is not None:
        exog_dim = exog_data.shape[-1] // series_num
        if series_num * exog_dim != exog_data.shape[-1]:
            raise ValueError("Exogenous columns cannot be evenly divided by series_num.")
        sample_exog_data = exog_data.iloc[:, :exog_dim]
        exog_data = adapter.reshape_spatiotemporal(exog_data, series_num)[0]
        train_valid_np = np.concatenate([train_valid_np, exog_data], axis=1)
        exog_dim = exog_data.shape[-2]
        sample_train_valid_data = pd.concat([sample_train_valid_data, sample_exog_data], axis=1)
    else:
        exog_dim = 0

    # Keep the existing hyper-parameter tuning utilities so channel counts and
    # downstream model metadata stay aligned with the normal benchmark path.
    if sample_train_valid_data.shape[1] == 1:
        adapter.single_forecasting_hyper_param_tune(sample_train_valid_data)
    else:
        adapter.multi_forecasting_hyper_param_tune(sample_train_valid_data)

    adapter.config.series_dim = series_dim
    adapter.config.input_dim = series_dim + exog_dim
    adapter.config.output_dim = series_dim
    adapter.config.adj_mx = adj_mx
    adapter.config.series_num = series_num
    adapter.config.num_input_channels = series_dim + exog_dim
    adapter.config.geo_data = kwargs.get("geo_data", None)
    if exog_dim > 0:
        adapter.config.prediction_channel_indices = list(range(series_dim))
        adapter.config.exogenous_channel_indices = list(range(series_dim, series_dim + exog_dim))
    else:
        adapter.config.prediction_channel_indices = None
        adapter.config.exogenous_channel_indices = None

    # Split the reshaped array with the same benchmark train/val rule used by
    # the regular adapter path.
    train_data, valid_data = train_val_split(
        train_valid_np, train_ratio_in_tv, adapter.config.seq_len
    )

    train_data_l = train_data.shape[0]

    # Fit target and exogenous scalers separately, mirroring the benchmark path.
    if exog_dim > 0:
        adapter.scaler1.fit(train_data[:, :series_dim, :].transpose(0, 2, 1).reshape(-1, series_dim))
        adapter.scaler2.fit(train_data[:, series_dim:, :].transpose(0, 2, 1).reshape(-1, exog_dim))
        if adapter.config.norm:
            scaled_series = adapter.scaler1.transform(
                train_data[:, :series_dim, :].transpose(0, 2, 1).reshape(-1, series_dim)
            )
            train_series = scaled_series.reshape(train_data_l, series_num, series_dim).transpose(0, 2, 1)
            scaled_exog = adapter.scaler2.transform(
                train_data[:, series_dim:, :].transpose(0, 2, 1).reshape(-1, exog_dim)
            )
            train_exog = scaled_exog.reshape(train_data_l, series_num, exog_dim).transpose(0, 2, 1)
            train_data = np.concatenate((train_series, train_exog), axis=1)
    else:
        adapter.scaler1.fit(train_data.transpose(0, 2, 1).reshape(-1, series_dim))
        if adapter.config.norm:
            scaled_train = adapter.scaler1.transform(
                train_data.transpose(0, 2, 1).reshape(-1, series_dim)
            )
            train_data = scaled_train.reshape(train_data_l, series_num, series_dim).transpose(0, 2, 1)

    if valid_data is not None and adapter.config.norm:
        valid_data_l = valid_data.shape[0]
        if exog_dim > 0:
            scaled_series = adapter.scaler1.transform(
                valid_data[:, :series_dim, :].transpose(0, 2, 1).reshape(-1, series_dim)
            )
            valid_series = scaled_series.reshape(valid_data_l, series_num, series_dim).transpose(0, 2, 1)
            scaled_exog = adapter.scaler2.transform(
                valid_data[:, series_dim:, :].transpose(0, 2, 1).reshape(-1, exog_dim)
            )
            valid_exog = scaled_exog.reshape(valid_data_l, series_num, exog_dim).transpose(0, 2, 1)
            valid_data = np.concatenate((valid_series, valid_exog), axis=1)
        else:
            scaled_valid = adapter.scaler1.transform(
                valid_data.transpose(0, 2, 1).reshape(-1, series_dim)
            )
            valid_data = scaled_valid.reshape(valid_data_l, series_num, series_dim).transpose(0, 2, 1)

    # TinyTimeMixerTrainerDataset will slide over time and then expand node N as
    # extra samples, effectively turning (T, C, N) into Trainer samples shaped
    # like (sample, past/future, C). When exog exists, those channels are kept in
    # the tensors; TTM's prediction/exogenous channel indices will ensure loss is
    # computed only on the target channels.
    train_dataset = TinyTimeMixerTrainerDataset(
        train_data,
        seq_len=adapter.config.seq_len,
        pred_len=adapter.config.horizon,
        series_num=series_num,
        use_frequency_token=bool(getattr(adapter.config, "use_frequency_token", True)),
        freq_token_id=int(getattr(adapter, "_ttm_freq_token_id", 0)),
    )

    # Few-shot reuses the benchmark subset helper after the trainer dataset has
    # already been materialized, so sampling happens over trainer-ready samples.
    if str(getattr(adapter.config, "shot_mode", "full_shot")).lower() == "few_shot":
        train_dataset = build_few_shot_subset(
            train_dataset,
            getattr(adapter.config, "sampling_rate", getattr(adapter.config, "few_shot_ratio", 0.1)),
            strategy=getattr(adapter.config, "sampling_strategy", "uniform"),
            seed=getattr(adapter.config, "seed", None),
        )

    valid_dataset = None
    if valid_data is not None:
        valid_dataset = TinyTimeMixerTrainerDataset(
            valid_data,
            seq_len=adapter.config.seq_len,
            pred_len=adapter.config.horizon,
            series_num=series_num,
            use_frequency_token=bool(getattr(adapter.config, "use_frequency_token", True)),
            freq_token_id=int(getattr(adapter, "_ttm_freq_token_id", 0)),
        )

    return train_dataset, valid_dataset


def _resolve_official_lr(model, train_dataset, config):

    official_lr = getattr(config, "official_learning_rate", None)
    base_lr = official_lr if official_lr is not None else getattr(config, "lr", 1e-4)
    learning_rate = float(base_lr)
    if not bool(getattr(config, "use_official_lr_finder", False)):
        return learning_rate, model

    granite_root = Path(__file__).resolve().parents[3] / "granite-tsfm-main"
    if granite_root.exists() and str(granite_root) not in sys.path:
        sys.path.insert(0, str(granite_root))

    from ts_benchmark.baselines.tsfm.utils.lr_finder import optimal_lr_finder


    learning_rate, model = optimal_lr_finder(
        model,
        train_dataset,
        batch_size=int(getattr(config, "batch_size", 32)),
        enable_prefix_tuning=bool(getattr(config, "use_frequency_token", True)),
    )
    return learning_rate, model


def run_ttm_official_trainer(model, train_dataset, valid_dataset, config):
    trainer_seed = int(getattr(config, "trainer_seed", getattr(config, "seed", 42)))
    set_seed(trainer_seed)
    os.environ.setdefault("WANDB_DISABLED", "true")

    learning_rate, model = _resolve_official_lr(model, train_dataset, config)

    tmp_dir = tempfile.mkdtemp()
    training_args_kwargs = {
        "output_dir": tmp_dir,
        "overwrite_output_dir": True,
        "learning_rate": learning_rate,
        "num_train_epochs": int(getattr(config, "num_epochs", 25)),
        "per_device_train_batch_size": int(getattr(config, "batch_size", 64)),
        "per_device_eval_batch_size": int(getattr(config, "batch_size", 64)),
        "dataloader_num_workers": int(getattr(config, "trainer_num_workers", 0)),
        "report_to": "none",
        "logging_strategy": "epoch",
        "logging_dir": tmp_dir,
        "seed": trainer_seed,
    }

    callbacks = []
    if valid_dataset is not None:
        training_args_kwargs.update(
            {
                "do_eval": True,
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            training_args_kwargs["eval_strategy"] = "epoch"
        else:
            training_args_kwargs["evaluation_strategy"] = "epoch"
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(getattr(config, "patience", 10)),
                early_stopping_threshold=0.0,
            )
        )
    else:
        training_args_kwargs.update(
            {
                "do_eval": False,
                "save_strategy": "no",
                "load_best_model_at_end": False,
            }
        )

    training_args = TrainingArguments(**training_args_kwargs)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=max(int(getattr(config, "num_epochs", 25)), 1),
        steps_per_epoch=max(math.ceil(len(train_dataset) / int(getattr(config, "batch_size", 64))), 1),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ttm_grouped_collate,
        callbacks=callbacks,
        optimizers=(optimizer, scheduler),
    )

    codecarbon_callback = INTEGRATION_TO_CALLBACK.get("codecarbon")
    if codecarbon_callback is not None:
        try:
            trainer.remove_callback(codecarbon_callback)
        except Exception:
            pass

    wandb_callback = INTEGRATION_TO_CALLBACK.get("wandb")
    if wandb_callback is not None:
        try:
            trainer.remove_callback(wandb_callback)
        except Exception:
            pass

    trainer.train()
    return trainer.model, learning_rate
