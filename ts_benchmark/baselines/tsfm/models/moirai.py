import sys
from pathlib import Path
from typing import Dict

import torch
from torch import nn

_SUBMODULE_ROOT = Path(__file__).resolve().parents[1] / "submodules"
if str(_SUBMODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE_ROOT))

from ts_benchmark.baselines.tsfm.submodules.uni2ts.model.moirai.module import MoiraiModule
from ts_benchmark.baselines.tsfm.submodules.uni2ts.model.moirai.forecast import MoiraiForecast


class Moirai(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.pred_len = int(getattr(configs, "pred_len", getattr(configs, "horizon", 96)))
        self.context_length = int(getattr(configs, "seq_len", 512))
        self.target_dim = int(getattr(configs, "target_dim", getattr(configs, "c_out", getattr(configs, "output_dim", 1))))
        self.patch_size = self._parse_patch_size(getattr(configs, "patch_size", 64))
        self.num_samples = int(getattr(configs, "num_samples", 100))
        self.freeze_backbone = bool(getattr(configs, "freeze_backbone", False))
        self.shot_mode = str(getattr(configs, "shot_mode", "full_shot")).lower()
        self.finetune_pattern = str(getattr(configs, "finetune_pattern", "full")).lower()

        pretrained_model_id = getattr(configs, "pretrained_model_id", "")
        model_size = getattr(configs, "model_size", "base")
        if not pretrained_model_id:
            pretrained_model_id = f"Salesforce/moirai-1.1-R-{model_size}"

        shared_module = MoiraiModule.from_pretrained(pretrained_model_id)

        # Forecast helper is kept for all shot modes:
        # - zero-shot inference
        # - formatting outputs and packed conversion in finetune modes
        self.forecast_model = MoiraiForecast(
            module=shared_module,
            prediction_length=self.pred_len,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=self.target_dim,
            num_samples=self.num_samples,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        self.finetune_model = None
        if self.shot_mode in {"few_shot", "full_shot"}:
            self.finetune_model = self._build_finetune_model(shared_module, configs)

        # Optional local checkpoint (not required for zero-shot)
        ckpt_path = getattr(configs, "pretrain_model_path", "")
        if ckpt_path:
            self._maybe_load_local_checkpoint(ckpt_path)

        self._apply_finetune_pattern()

    @staticmethod
    def _parse_patch_size(raw_patch_size):
        if isinstance(raw_patch_size, int):
            return raw_patch_size
        if isinstance(raw_patch_size, str):
            if raw_patch_size.lower() == "auto":
                return "auto"
            if raw_patch_size.isdigit():
                return int(raw_patch_size)
        return raw_patch_size

    def _build_finetune_model(self, shared_module: MoiraiModule, configs):
        try:
            from ts_benchmark.baselines.tsfm.submodules.uni2ts.model.moirai.finetune import (
                MoiraiFinetune,
            )
        except Exception as exc:
            raise ImportError(
                "Failed to import MoiraiFinetune for few/full-shot. "
                "Please ensure uni2ts finetune dependencies are installed (e.g. lightning)."
            ) from exc

        patch_size = self.forecast_model.hparams.patch_size
        if patch_size == "auto":
            raise ValueError(
                "Official finetune mode requires a fixed integer patch_size; got patch_size='auto'."
            )

        return MoiraiFinetune(
            min_patches=int(getattr(configs, "min_patches", 2)),
            min_mask_ratio=float(getattr(configs, "min_mask_ratio", 0.15)),
            max_mask_ratio=float(getattr(configs, "max_mask_ratio", 0.5)),
            max_dim=int(getattr(configs, "max_dim", 128)),
            num_training_steps=getattr(configs, "num_training_steps", None),
            num_warmup_steps=int(getattr(configs, "num_warmup_steps", 0)),
            module=shared_module,
            num_samples=self.num_samples,
            context_length=self.context_length,
            prediction_length=self.pred_len,
            patch_size=int(patch_size),
            finetune_pattern=self.finetune_pattern,
        )

    def _maybe_load_local_checkpoint(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        if not isinstance(ckpt, dict):
            return

        prefixes = (
            "module.",
            "model.",
            "moirai.",
            "forecast_model.",
            "finetune_model.",
        )

        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in ckpt.items():
            kk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if kk.startswith(p):
                        kk = kk[len(p):]
                        changed = True
            cleaned[kk] = v

        # Load into shared MoiraiModule (official pretrained backbone)
        self.forecast_model.module.load_state_dict(cleaned, strict=False)

    def _apply_finetune_pattern(self):
        if self.freeze_backbone or self.shot_mode == "zero_shot":
            for p in self.parameters():
                p.requires_grad = False
            return

        if self.finetune_pattern == "full":
            return

        if self.finetune_pattern == "freeze_ffn":
            for pn, p in self.named_parameters():
                if "ffn" in pn:
                    p.requires_grad = False
            return

        if self.finetune_pattern == "head_only":
            for pn, p in self.named_parameters():
                if "param_proj" not in pn:
                    p.requires_grad = False
            return

        raise ValueError(f"Unsupported finetune_pattern: {self.finetune_pattern}")

    @staticmethod
    def _collapse_sample_forecasts(forecasts: torch.Tensor) -> torch.Tensor:
        sorted_forecasts = torch.sort(forecasts, dim=1).values
        quantile_idx = int(round((sorted_forecasts.shape[1] - 1) * 0.5))

        if forecasts.dim() == 4:
            return sorted_forecasts[:, quantile_idx, :, :]
        if forecasts.dim() == 3:
            return sorted_forecasts[:, quantile_idx, :].unsqueeze(-1)

        raise ValueError(f"Unexpected Moirai forecast shape: {tuple(forecasts.shape)}")

    def _forecast_inference(self, x_enc: torch.Tensor) -> torch.Tensor:
        device = x_enc.device
        past_is_pad = torch.zeros((x_enc.shape[0], x_enc.shape[1]), dtype=torch.bool, device=device)
        past_observed_target = torch.ones_like(x_enc, dtype=torch.bool, device=device)
        forecasts = self.forecast_model(
            past_target=x_enc,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            num_samples=self.num_samples,
        )

        pred = self._collapse_sample_forecasts(forecasts)

        return pred[:, -self.pred_len:, :]

    def forward_train(self, x_enc: torch.Tensor, target: torch.Tensor):
        """
        Official-like finetune path:
        - build packed tensors with MoiraiForecast._convert
        - run MoiraiFinetune.forward to get distribution
        - optimize PackedNLLLoss on prediction window
        """
        if self.finetune_model is None:
            pred = self._forecast_inference(x_enc)
            return pred, x_enc.new_zeros(())

        if x_enc.dim() != 3 or target.dim() != 3:
            raise ValueError(
                f"Moirai finetune expects 3D tensors [B,T,C], got x_enc={tuple(x_enc.shape)}, target={tuple(target.shape)}"
            )

        future = target[:, -self.pred_len :, :]
        tgt_dim = min(self.target_dim, x_enc.shape[-1], future.shape[-1])

        past_target = x_enc[..., :tgt_dim]
        future_target = future[..., :tgt_dim]

        # Robust to possible NaN in data
        past_observed_target = torch.isfinite(past_target)
        future_observed_target = torch.isfinite(future_target)
        past_target = torch.nan_to_num(past_target, nan=0.0, posinf=0.0, neginf=0.0)
        future_target = torch.nan_to_num(future_target, nan=0.0, posinf=0.0, neginf=0.0)

        past_is_pad = torch.zeros((past_target.shape[0], past_target.shape[1]), dtype=torch.bool, device=past_target.device)
        future_is_pad = torch.zeros((future_target.shape[0], future_target.shape[1]), dtype=torch.bool, device=future_target.device)

        patch_size = self.forecast_model.hparams.patch_size
        if patch_size == "auto":
            raise ValueError(
                "Official finetune mode requires a fixed integer patch_size; got patch_size='auto'."
            )
        patch_size = int(patch_size)

        (
            packed_target,
            packed_observed,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
        ) = self.forecast_model._convert(
            patch_size=patch_size,
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            future_target=future_target,
            future_observed_target=future_observed_target,
            future_is_pad=future_is_pad,
        )

        patch_size_tensor = torch.ones_like(time_id, dtype=torch.long) * patch_size

        distr = self.finetune_model(
            target=packed_target,
            observed_mask=packed_observed,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            patch_size=patch_size_tensor,
        )

        nll_loss = self.finetune_model.hparams.loss_func(
            pred=distr,
            target=packed_target,
            prediction_mask=prediction_mask,
            observed_mask=packed_observed,
            sample_id=sample_id,
            variate_id=variate_id,
        )

        pred = self.forecast_model._format_preds(
            patch_size,
            distr.mean.unsqueeze(0),
            tgt_dim,
        )

        pred = self._collapse_sample_forecasts(pred)

        return pred, nll_loss

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self._forecast_inference(x_enc)
