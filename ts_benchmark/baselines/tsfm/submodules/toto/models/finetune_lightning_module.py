from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.optim import AdamW

from ts_benchmark.baselines.tsfm.submodules.toto.dataset_utils import CausalMaskedTimeseries
from ts_benchmark.baselines.tsfm.submodules.toto.models.backbone import TotoBackbone, TotoOutput
from ts_benchmark.baselines.tsfm.submodules.toto.models.finetune_losses import CombinedLoss
from ts_benchmark.baselines.tsfm.submodules.toto.models.finetune_scheduler import WarmupStableDecayLR


class TotoForFinetuning(L.LightningModule):
    def __init__(
        self,
        val_prediction_len: int = 96,
        stable_steps: int = 1000,
        decay_steps: int = 1000,
        warmup_steps: int = 200,
        lr: float = 1e-4,
        min_lr: float = 1e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        pretrained_backbone: TotoBackbone | None = None,
        add_exogenous_features: bool = False,
        **model_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_backbone"])

        if pretrained_backbone is not None:
            self.model = pretrained_backbone
        else:
            self.model = TotoBackbone(**model_kwargs)

        if add_exogenous_features:
            self.model.enable_variate_labels()

        self.lr = lr
        self.min_lr = min_lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.val_prediction_len = val_prediction_len

        self.combined_loss = CombinedLoss()

    def configure_optimizers(self):
        decay_params = [param for param in self.parameters() if param.requires_grad]

        optimizer = AdamW(
            decay_params,
            lr=self.lr,
            betas=self.betas,
            eps=1e-7,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = WarmupStableDecayLR(
            optimizer=optimizer,
            warmup_steps=self.warmup_steps,
            stable_steps=self.stable_steps,
            decay_steps=self.decay_steps,
            min_lr=self.min_lr,
            base_lr=self.lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _prediction_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        pred_len = min(self.val_prediction_len, padding_mask.shape[-1])
        mask = torch.zeros_like(padding_mask, dtype=torch.bool)
        if pred_len > 0:
            mask[..., -pred_len:] = True
        return mask & padding_mask.bool()

    def _get_inputs(self, x: CausalMaskedTimeseries) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        input_slice = x.input_slice
        return (
            x.series[..., input_slice],
            x.padding_mask[..., input_slice],
            x.id_mask[..., input_slice],
            x.num_exogenous_variables,
        )

    def forward(self, x: CausalMaskedTimeseries) -> TotoOutput:
        inputs, input_padding_mask, id_mask, num_exogenous_variables = self._get_inputs(x)
        return self.model(
            inputs,
            input_padding_mask,
            id_mask,
            num_exogenous_variables=num_exogenous_variables,
        )

    def _train_or_val_step(self, batch: CausalMaskedTimeseries, is_train: bool) -> torch.Tensor:
        eps = torch.finfo(batch.series.dtype).eps
        target_slice = batch.target_slice
        targets = batch.series[..., target_slice]
        targets_padding_mask = batch.padding_mask[..., target_slice]

        out = self(batch)
        distr, loc, scale = out.distribution, out.loc, out.scale
        scaled_targets = (targets - loc) / (scale + eps)

        if not is_train:
            mask = self._prediction_mask(targets_padding_mask)
        else:
            mask = targets_padding_mask

        if batch.num_exogenous_variables > 0:
            mask[..., -batch.num_exogenous_variables :] = False

        loss = self.combined_loss(distr, scaled_targets) * mask

        valid_count = mask.sum()
        total_sum = loss.sum()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(valid_count, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_sum, op=torch.distributed.ReduceOp.SUM)

        mean_total = total_sum / (valid_count + eps)
        prefix = "train" if is_train else "val"

        self.log(
            f"{prefix}_loss",
            mean_total,
            prog_bar=True,
            on_step=is_train,
            on_epoch=True,
            batch_size=batch.series.shape[0],
        )
        return mean_total

    def training_step(self, batch: CausalMaskedTimeseries, _batch_idx: int):
        return self._train_or_val_step(batch, is_train=True)

    def validation_step(self, batch: CausalMaskedTimeseries, _batch_idx: int):
        return self._train_or_val_step(batch, is_train=False)
