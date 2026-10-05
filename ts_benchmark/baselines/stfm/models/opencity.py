"""Thin benchmark wrapper around the vendored OpenCity model."""

from __future__ import annotations

import os
from typing import Mapping

import torch
from torch import nn

from ts_benchmark.baselines.stfm.models.opencity_io import (
    OPENCITY_CHECKPOINT,
    OPENCITY_CONTEXT,
    OPENCITY_DATASET_KEY,
    OPENCITY_HORIZON,
    build_opencity_args,
)
from ts_benchmark.baselines.stfm.submodules.opencity.OpenCity import (
    OpenCity as VendoredOpenCity,
)


def _get_config_value(configs, *names, default=None):
    for name in names:
        if hasattr(configs, name):
            return getattr(configs, name)
    return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _strip_checkpoint_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict:
    cleaned = {}
    prefixes = (
        "module.predictor.",
        "predictor.",
        "module.model.",
        "model.",
        "module.",
    )
    for key, value in state_dict.items():
        clean_key = key
        for prefix in prefixes:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                break
        cleaned[clean_key] = value
    return cleaned


class OpenCity_STFM(nn.Module):
    """Benchmark-compatible wrapper that preserves OpenCity's 288->288 geometry."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.dataset_key = str(
            _get_config_value(
                configs,
                "opencity_dataset",
                "dataset",
                "data_name",
                default=OPENCITY_DATASET_KEY,
            )
        )
        self.input_window = OPENCITY_CONTEXT
        self.output_window = OPENCITY_HORIZON
        self.output_dim = int(_get_config_value(configs, "series_dim", "output_dim", default=1))
        if self.output_dim != 1:
            raise ValueError(
                f"OpenCity_STFM currently supports one target channel, got {self.output_dim}."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = build_opencity_args(configs, dataset_key=self.dataset_key)
        self.model = VendoredOpenCity(
            args=args,
            dataset_use=[self.dataset_key],
            device=device,
            dim_in=self.output_dim,
        )

        mode = str(getattr(configs, "shot_mode", "full_shot")).lower()
        ckpt_path = str(
            _get_config_value(
                configs,
                "pretrain_model_path",
                "ckpt_path",
                "opencity_checkpoint_path",
                default=OPENCITY_CHECKPOINT,
            )
            or ""
        ).strip()
        self.loaded_checkpoint_path = ""
        if mode != "ori" and ckpt_path:
            self.load_backbone_weights(ckpt_path)

        freeze_backbone = _as_bool(getattr(configs, "freeze_backbone", mode != "ori"))
        if mode in {"few_shot", "full_shot"} and freeze_backbone:
            self.freeze_except_prediction_head()
        elif mode == "zero_shot":
            self.freeze_all()

    def load_backbone_weights(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"OpenCity checkpoint not found: {path}")
        state_dict = torch.load(path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        cleaned = _strip_checkpoint_prefix(state_dict)
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        critical_missing = [key for key in missing if not key.startswith("sem_mask")]
        if critical_missing or unexpected:
            raise RuntimeError(
                "OpenCity checkpoint load mismatch: "
                f"missing={critical_missing[:10]}, unexpected={unexpected[:10]}"
            )
        self.loaded_checkpoint_path = path

    def freeze_all(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def freeze_except_prediction_head(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.linear.parameters():
            param.requires_grad = True

    def forward(self, x_open, y_open):
        self.model.device = x_open.device
        return self.model(x_open, y_open, self.dataset_key)
