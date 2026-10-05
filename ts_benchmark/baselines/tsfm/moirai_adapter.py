from typing import Type

import torch
import torch.nn as nn

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    MOIRAI_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.tsfm.moirai_trainer_helpers import (
    run_moirai_official_trainer,
)


class MoiraiAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, MOIRAI_HPARAMS, **kwargs)

    def _shot_mode(self) -> str:
        return str(getattr(self.config, "shot_mode", "full_shot")).lower()

    def _is_official_finetune_mode(self) -> bool:
        return self._shot_mode() in {"few_shot", "full_shot"}

    def _uses_official_trainer(self) -> bool:
        return self._is_official_finetune_mode() and str(
            getattr(self.config, "finetune_style", "official_trainer")
        ).lower() == "official_trainer"

    def _uses_internal_train_loss(self) -> bool:
        return self._is_official_finetune_mode() and not self._uses_official_trainer()

    def _on_shot_mode_resolved(self, mode: str) -> None:
        if mode in {"few_shot", "full_shot"}:
            self.config.lradj = "TST"

    def _init_optimizer(self, CovariateFusion=None):
        if not self._is_official_finetune_mode() or self._uses_official_trainer():
            return super()._init_optimizer(CovariateFusion=CovariateFusion)

        base_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        finetune_model = getattr(base_model, "finetune_model", None)
        if finetune_model is None:
            return super()._init_optimizer(CovariateFusion=CovariateFusion)

        whitelist_names = {"LearnedProjection", "MultiInSizeLinear", "MultiOutSizeLinear", "Linear"}
        blacklist_names = {"BinaryAttentionBias", "LearnedEmbedding", "RMSNorm", "Embedding", "LayerNorm"}

        decay = set()
        no_decay = set()

        for mn, m in finetune_model.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                fpn = f"{mn}.{pn}" if mn else pn
                cls_name = m.__class__.__name__
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and cls_name in whitelist_names:
                    decay.add(fpn)
                elif pn.endswith("weight") and cls_name in blacklist_names:
                    no_decay.add(fpn)
                else:
                    no_decay.add(fpn)

        param_dict = {
            pn: p
            for pn, p in finetune_model.named_parameters()
            if p.requires_grad
        }
        missing = set(param_dict.keys()) - (decay | no_decay)
        no_decay |= missing

        optim_groups = []
        if decay:
            optim_groups.append(
                {
                    "params": [param_dict[pn] for pn in sorted(decay)],
                    "weight_decay": float(getattr(self.config, "weight_decay", 1e-1)),
                }
            )
        if no_decay:
            optim_groups.append(
                {
                    "params": [param_dict[pn] for pn in sorted(no_decay)],
                    "weight_decay": 0.0,
                }
            )

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=float(getattr(self.config, "lr", 5e-7)),
            betas=(
                float(getattr(self.config, "beta1", 0.9)),
                float(getattr(self.config, "beta2", 0.98)),
            ),
            eps=1e-6,
        )

        steps_per_epoch = max(1, len(self.train_data_loader))
        total_steps = getattr(self.config, "num_training_steps", None)
        if total_steps is None:
            total_steps = steps_per_epoch * max(1, int(getattr(self.config, "num_epochs", 1)))
        warmup_steps = int(getattr(self.config, "num_warmup_steps", 0))

        if warmup_steps > 0:
            def lr_lambda(step):
                return min(1.0, float(step + 1) / float(max(1, warmup_steps)))
        else:
            def lr_lambda(step):
                return 1.0

        self._official_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer

    def _adjust_lr(self, optimizer, epoch, config):
        if self._is_official_finetune_mode() and not self._uses_official_trainer():
            if self._official_scheduler is not None:
                self._official_scheduler.step()
            return
        return super()._adjust_lr(optimizer, epoch, config)

    def _forecast_fit_official_trainer(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        self.model = self._init_model()
        self.CovariateFusion = None
        self.check_point = None

        result = run_moirai_official_trainer(
            self,
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )

        self.check_point = self.save_checkpoint({"Model": self.model})
        print(
            "Moirai official_trainer finished: "
            f"train_windows={result.train_windows}, "
            f"val_windows={result.val_windows}, "
            f"best_val_packed_nll={result.best_val_packed_nll}, "
            f"best_ckpt_path={result.best_ckpt_path}, "
            f"run_dir={result.run_dir}, "
            f"debug_dir={result.debug_dir}"
        )
        return self

    def forecast_fit(
        self,
        train_valid_data,
        *,
        covariates=None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        mode = self._resolve_shot_mode()
        self._setup_post_norm_shot(mode)

        if self._uses_official_trainer():
            return self._forecast_fit_official_trainer(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )

        return BaseTSFMAdapter.forecast_fit(
            self,
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )


def moirai_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=MoiraiAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )


def tsfm_adapter(model_info: Type[object]) -> object:
    return moirai_adapter(model_info)
