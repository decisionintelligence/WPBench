from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from einops import rearrange
from torch.utils.data import DataLoader, RandomSampler

from ts_benchmark.baselines.tsfm.few_shot_utils import build_few_shot_subset
from ts_benchmark.baselines.tsfm.submodules.toto.dataset_utils import (
    pad_array,
    pad_id_mask,
)
from ts_benchmark.baselines.tsfm.submodules.toto.models.finetune_losses import (
    CombinedLoss,
)


@dataclass(frozen=True)
class TotoFinetuneBatch:
    model_inputs_bvt: torch.Tensor
    input_padding_mask_bvt: torch.Tensor
    id_mask_bvt: torch.Tensor
    shifted_targets_bvt: torch.Tensor
    loss_mask_bvt: torch.Tensor
    target_dim: int
    num_exogenous_variables: int
    original_input_length: int


def flatten_benchmark_tensor(tensor: torch.Tensor, tensor_name: str) -> torch.Tensor:
    if tensor.dim() == 4:
        return rearrange(tensor, "b n t c -> (b n) t c")
    if tensor.dim() == 3:
        return tensor
    raise ValueError(
        f"Toto finetuning expects {tensor_name} shaped [B,T,C] or [B,N,T,C], "
        f"got {tuple(tensor.shape)}."
    )


def ensure_zero_shot_mode(adapter_cfg: Any) -> None:
    if adapter_cfg is None:
        raise ValueError("adapter_cfg must not be None.")

    adapter_cfg.shot_mode = "zero_shot"
    adapter_cfg.num_epochs = 0
    adapter_cfg.freeze_backbone = True
    adapter_cfg.finetune_style = "benchmark"


def build_finetune_loss(adapter_cfg: Any) -> CombinedLoss:
    loss_name = str(getattr(adapter_cfg, "toto_loss_name", "combined")).lower()
    if loss_name != "combined":
        raise ValueError(f"Unsupported Toto finetune loss: {loss_name}")

    return CombinedLoss(
        lambda_nll=float(getattr(adapter_cfg, "toto_loss_lambda_nll", 0.575)),
        delta=float(getattr(adapter_cfg, "toto_loss_delta", 0.1)),
        alpha=float(getattr(adapter_cfg, "toto_loss_alpha", 0.0)),
    )


def prepare_finetune_batch(
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    *,
    horizon: int,
    target_dim: int,
    patch_size: int,
    exclude_exogenous_dimensions: bool = False,
) -> TotoFinetuneBatch:
    input_btc = flatten_benchmark_tensor(input_tensor, "input")
    target_btc = flatten_benchmark_tensor(target_tensor, "target")

    if input_btc.shape[0] != target_btc.shape[0]:
        raise ValueError(
            "Toto finetune batch mismatch between input and target batch sizes: "
            f"input={tuple(input_btc.shape)}, target={tuple(target_btc.shape)}."
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if horizon > input_btc.shape[1]:
        raise ValueError(
            "Toto finetune horizon cannot exceed input length: "
            f"horizon={horizon}, input_len={input_btc.shape[1]}."
        )
    if horizon > target_btc.shape[1]:
        raise ValueError(
            "Toto finetune horizon cannot exceed target length: "
            f"horizon={horizon}, target_len={target_btc.shape[1]}."
        )
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}.")
    if patch_size > input_btc.shape[1]:
        raise ValueError(
            "Toto finetune patch_size cannot exceed input length: "
            f"patch_size={patch_size}, input_len={input_btc.shape[1]}."
        )
    if patch_size > horizon:
        raise ValueError(
            "Toto finetune patch_size cannot exceed prediction horizon: "
            f"patch_size={patch_size}, horizon={horizon}."
        )
    if target_dim <= 0 or target_dim > input_btc.shape[-1]:
        raise ValueError(
            "Toto finetune target_dim must be within the input channel range: "
            f"target_dim={target_dim}, channels={input_btc.shape[-1]}."
        )
    if target_dim > target_btc.shape[-1]:
        raise ValueError(
            "Toto finetune target_dim cannot exceed target channels: "
            f"target_dim={target_dim}, target_channels={target_btc.shape[-1]}."
        )

    batch_size, input_len, input_channels = input_btc.shape
    num_exogenous_variables = int(input_channels - target_dim)

    future_horizon_btc = target_btc[:, -horizon:, :target_dim]
    future_patch_btc = future_horizon_btc[:, :patch_size, :]
    future_tail = input_btc.new_zeros((batch_size, patch_size, input_channels))
    future_tail[:, :, :target_dim] = future_patch_btc

    shifted_targets_btc = torch.cat([input_btc[:, patch_size:, :], future_tail], dim=1)
    model_inputs_bvt = input_btc.permute(0, 2, 1).contiguous()
    shifted_targets_bvt = shifted_targets_btc.permute(0, 2, 1).contiguous()

    input_padding_mask_bvt = torch.ones_like(model_inputs_bvt, dtype=torch.bool)
    loss_mask_bvt = torch.ones_like(shifted_targets_bvt, dtype=torch.bool)
    if num_exogenous_variables > 0:
        del exclude_exogenous_dimensions
        loss_mask_bvt[:, target_dim:, :] = False

    id_mask_bvt = torch.zeros(
        model_inputs_bvt.shape,
        dtype=torch.int,
        device=model_inputs_bvt.device,
    )

    if model_inputs_bvt.shape[-1] % patch_size == 0:
        padded_inputs_bvt = model_inputs_bvt
        padded_input_padding_mask_bvt = input_padding_mask_bvt
        padded_shifted_targets_bvt = shifted_targets_bvt
        padded_loss_mask_bvt = loss_mask_bvt
        padded_id_mask_bvt = id_mask_bvt
    else:
        padded_inputs_bvt = pad_array(model_inputs_bvt, patch_size)
        padded_input_padding_mask_bvt = pad_array(input_padding_mask_bvt, patch_size)
        padded_shifted_targets_bvt = pad_array(shifted_targets_bvt, patch_size)
        padded_loss_mask_bvt = pad_array(loss_mask_bvt, patch_size)
        padded_id_mask_bvt = pad_id_mask(id_mask_bvt, patch_size)

    return TotoFinetuneBatch(
        model_inputs_bvt=padded_inputs_bvt,
        input_padding_mask_bvt=padded_input_padding_mask_bvt,
        id_mask_bvt=padded_id_mask_bvt,
        shifted_targets_bvt=padded_shifted_targets_bvt,
        loss_mask_bvt=padded_loss_mask_bvt,
        target_dim=target_dim,
        num_exogenous_variables=num_exogenous_variables,
        original_input_length=input_len,
    )


def compute_finetune_loss(
    loss_fn: CombinedLoss,
    distribution: torch.distributions.Distribution,
    loc: torch.Tensor,
    scale: torch.Tensor,
    batch: TotoFinetuneBatch,
    *,
    prediction_mask_length: int | None = None,
) -> torch.Tensor:
    eps = torch.finfo(batch.shifted_targets_bvt.dtype).eps
    scaled_targets = (batch.shifted_targets_bvt - loc) / (scale + eps)
    pointwise_loss = loss_fn(distribution, scaled_targets)
    loss_mask = batch.loss_mask_bvt.to(dtype=pointwise_loss.dtype)
    if prediction_mask_length is not None:
        pred_len = max(0, min(int(prediction_mask_length), loss_mask.shape[-1]))
        prediction_mask = torch.zeros_like(loss_mask)
        if pred_len > 0:
            prediction_mask[..., -pred_len:] = 1
        loss_mask = loss_mask * prediction_mask
    masked_loss = pointwise_loss * loss_mask
    valid_count = loss_mask.sum()
    return masked_loss.sum() / (valid_count + eps)


def distribution_mean_to_bvt(
    distribution: torch.distributions.Distribution,
    loc: torch.Tensor,
    scale: torch.Tensor,
    *,
    output_length: int,
) -> torch.Tensor:
    mean_bvt = distribution.mean
    if loc.shape[-1] == 1:
        restored = mean_bvt * scale + loc
    else:
        restored = mean_bvt * scale[..., -mean_bvt.shape[-1] :] + loc[..., -mean_bvt.shape[-1] :]
    return restored[..., -output_length:]


def _resolve_toto_num_train_samples(adapter_cfg: Any, available_windows: int) -> int:
    raw = getattr(adapter_cfg, "toto_num_train_samples", getattr(adapter_cfg, "num_train_samples", 100))
    requested = int(raw) if raw is not None else 100
    if requested <= 0:
        requested = int(available_windows)
    return max(1, min(int(available_windows), requested))


def apply_toto_training_loader(
    dataset,
    loader,
    config,
    *,
    drop_last: bool,
    collate_fn=None,
):
    mode = str(getattr(config, "shot_mode", "full_shot")).lower()
    if mode not in {"few_shot", "zero_shot", "full_shot"}:
        raise ValueError(f"Unknown shot_mode: {mode}")

    if mode == "zero_shot":
        return loader

    sampled_dataset = dataset
    if mode == "few_shot":
        sampled_dataset = build_few_shot_subset(
            dataset,
            getattr(config, "sampling_rate", getattr(config, "few_shot_ratio", 0.1)),
            strategy=getattr(config, "sampling_strategy", "uniform"),
            seed=getattr(config, "seed", None),
        )

    num_samples = _resolve_toto_num_train_samples(config, len(sampled_dataset))
    sampler = RandomSampler(
        sampled_dataset,
        replacement=False,
        num_samples=num_samples,
    )
    return DataLoader(
        sampled_dataset,
        batch_size=getattr(config, "batch_size", loader.batch_size),
        sampler=sampler,
        num_workers=getattr(config, "num_workers", 0),
        drop_last=drop_last,
        collate_fn=collate_fn if collate_fn is not None else loader.collate_fn,
    )
