from typing import Dict, Optional, Literal

import pandas as pd
import torch
import torch.nn as nn

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.tsfm.few_shot_utils import apply_few_shot_loader

GENERIC_TSFM_HPARAMS = {
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 10,
    "patience": 3,
    "label_len": 0,
    "norm": False,
    "shot_mode": "full_shot",
    "few_shot_ratio": 0.1,
    "few_shot_strategy": "auto", 
    "sampling_basis": "sample",
    "sampling_strategy": "uniform",
    "freeze_backbone": False,
    "pretrain_model_path": "",
    "ckpt_path": "",
}

MOIRAI_HPARAMS = {
    **GENERIC_TSFM_HPARAMS,
    "lr": 5e-7,
    "num_epochs": 200,
    "finetune_pattern": "full",
    "pretrained_model_id": "Salesforce/moirai-1.1-R-base",
    "model_size": "base",
    "patch_size": 64,
    "num_samples": 100,
    "target_dim": 1,
    "min_patches": 2,
    "min_mask_ratio": 0.15,
    "max_mask_ratio": 0.5,
    "max_dim": 128,
    "beta1": 0.9,
    "beta2": 0.98,
    "weight_decay": 1e-1,
    "num_warmup_steps": 0,
    "num_training_steps": None,
    "few_shot_strategy": "auto", 
}

TIMER_HPARAMS = {
    "task_name": "forecast",
    "freq": "h",
    "batch_size": 32,
    "num_workers": 0,
    "lr": 1e-4,
    "num_epochs": 10,
    "patience": 3,
    "norm": True,
    "shot_mode": "full_shot",
    "few_shot_ratio": 0.1,
    "freeze_backbone": False,
    "pretrain_model_path": "",
    "ckpt_path": "",
    "patch_len": 24,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "factor": 1,
    "dropout": 0.1,
    "activation": "gelu",
    "output_attention": False,
    "use_ims": True,
    "label_len": -1,
    "decay_fac": 0.75,
    "lradj": "type1",
    "use_weight_decay": 0,
    "weight_decay": 0.01,
    "cos_max_decay_epoch": 10,
    "is_spatial": False,
    "channel_independence": True,
    "sampling_basis": "sample",
    "few_shot_strategy": "auto", 
    "external_scaler_mode": "pooled_target",
    "use_ims": True,
}

TIME_LLM_HPARAMS = {
    'task_name': 'long_term_forecast',
    'pred_len': 24,
    'seq_len': 96,
    'label_len': 48,
    'd_ff': 32,
    'patch_len': 16,
    'stride': 8,
    'freq': 'h',
    'dropout': 0.1,
    'patience': 10,
    'loss': 'MSE',
    'lradj': 'type1',
    'itr': 1,
    'lr': 0.0001,
    'num_workers': 10,
    'num_epochs': 10,
    'align_epochs': 10,
    'batch_size': 32,
    'activation': 'gelu',
    'd_layers': 1,
    'e_layers': 2,
    'enc_in': 7,
    'dec_in': 7,
    'c_out': 7,  #output size
    'n_heads': 8,
    'd_model': 16,
    'factor': 1,
    'embed': 'timeF',
    'output_attention': True,
    'prompt_domain': 0,
    'llm_model': 'GPT2',
    'llm_dim': 4096,
    'use_amp': 0,
}

TTM_HPARAMS = {
    **GENERIC_TSFM_HPARAMS,
    "freq": "h",
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 10,
    "patience": 3,
    "norm": True,
    "seq_len": 512,
    "label_len": 0,
    "target_dim": 1,
    "ttm_model_card": "ibm-granite/granite-timeseries-ttm-r2",
    "ttm_yaml_path": "",
    "ttm_checkpoint_path": "",
    "ttm_context_length": 512,
    "prefix_horizon_slice": True,
    "finetune_style": "benchmark",
    "freeze_backbone_in_finetune": True,
    "official_learning_rate": None,
    "official_batch_size": 64,
    "official_num_epochs": 25,
    "official_patience": 10,
    "head_dropout": 0.2,
    "use_frequency_token": True,
    "use_official_lr_finder": False,
    "trainer_seed": 42,
    "trainer_num_workers": 0,
}

SEMPO_HPARAMS = {
    "freq": "h",
    "batch_size": 128,
    "lr": 1e-4,
    "num_epochs": 20,
    "patience": 6,
    "norm": True,
    "shot_mode": "full_shot",
    "few_shot_ratio": 0.1,
    "sampling_strategy": "uniform",
    "freeze_backbone": True,
    "pretrain_model_path": "",
    "ckpt_path": "",
    "seq_len": 512,
    "label_len": 48,
    "patch_len": 64,
    "stride": 64,
    "d_model": 256,
    "e_layers": 3,
    "d_layers": 3,
    "domain_len": 128,
    "horizon_lengths": [1, 96, 192, 336, 720],
    "c_in": 1,
    "head_type": "prediction",
    "external_scaler_mode": "pooled_target",
}

TOTO_HPARAMS = {
    'batch_size': 1,
    'norm': True,
    'seed': 59,
    'deterministic_algorithms': True,
    'samples_per_batch': 256,
    'checkpoint_path': 'Datadog/Toto-Open-Base-1.0',
    'finetune_style': 'benchmark',
    'toto_official_train_batch_size': None,
    'toto_official_val_batch_size': 1,
    'toto_official_num_workers': 0,
    'toto_official_num_train_samples': None,
    'toto_official_max_steps': None,
    'toto_official_log_every_n_steps': 1,
    'toto_official_num_sanity_val_steps': 0,
    'toto_official_enable_progress_bar': True,
    'toto_official_refresh_rate': 1,
    'toto_official_val_check_interval': None,
    'toto_official_lr': None,
    'toto_official_min_lr': None,
    'toto_official_warmup_steps': None,
    'toto_official_stable_steps': None,
    'toto_official_decay_steps': None,
    'context-lengths': [2048],
    'data_split': 'test',
    'eval_stride': 512,
    'num_samples': 256,
    'prediction_length': [96, 192, 336, 720],
    'use_kv_cache': True,
    'dropout': 0.1,
    'embed_dim': 768,
    'mlp_hidden_dim': 3072,
    'num_heads': 12,
    'num_layers': 12,
    'patch_size': 64,
    'stride': 64,
    'spacewise_every_n_layers': 12,
    'spacewise_first': False,
    'scale_factor_exponent': 10.0,
    'stabilize_with_global': True,
    'use_memory_efficient_attention': False,
    'scaler_cls': "<class 'model.scaler.CausalPatchStdMeanScaler'>",
    'output_distribution_classes': ["<class 'model.distribution.MixtureOfStudentTsOutput'>"],
    "output_distribution_kwargs": {"k_components": 1,},
    "toto_num_train_samples": 100,
    "toto_loss_name": "combined",
    "toto_loss_lambda_nll": 0.575,
    "toto_loss_delta": 0.1,
    "toto_loss_alpha": 0.0,
    "toto_point_loss_weight": 1.0,
}

TIMESFM_HPARAMS = {
    **GENERIC_TSFM_HPARAMS,
    "backend": "gpu" if torch.cuda.is_available() else "cpu",
    "timesfm_repo_id": "google/timesfm-1.0-200m-pytorch",
    "timesfm_checkpoint_path": "",
    "timesfm_context_len": 512,
    "timesfm_horizon_len": 128,
    "timesfm_input_patch_len": 32,
    "timesfm_output_patch_len": 128,
    "timesfm_num_layers": 20,
    "timesfm_num_heads": 16,
    "timesfm_model_dims": 1280,
    "timesfm_per_core_batch_size": 32,
    "timesfm_point_forecast_mode": "median",
    "timesfm_freq": "h",
    "timesfm_freq_type": 0,
    "timesfm_use_quantile_loss": False,
    "timesfm_use_official_finetune": True,
    "freeze_backbone": False,
    "external_scaler_mode": "pooled_target",
}

class _ZeroLoss(nn.Module):
    def forward(self, pred, target):
        return pred.new_zeros(())


class TSFMAdapterWorkflowBase:

    def _on_shot_mode_resolved(self, mode: str) -> None:
        del mode
    
    def _setup_post_norm_shot(self, mode: str) -> None:
        if mode == "few_shot":
            # 设置shot_ratio和few_shot_ratio
            self._sampling_rate()
            # 设置（前/后/随机/均匀）采样策略
            self._sampling_strategy()
            self.config.sampling_basis = "sample"

            self._sampling_basis()
            return
        if mode in {"zero_shot", "full_shot"}:
            return
        raise ValueError(f"Unknown shot_mode: {mode}")


    def _resolve_shot_mode(self) -> str:
        mode = str(getattr(self.config, "shot_mode", "full_shot")).lower()
        if mode not in {"few_shot", "zero_shot", "full_shot"}:
            raise ValueError(f"Unknown shot_mode: {mode}")

        self.config.shot_mode = mode
        self._on_shot_mode_resolved(mode)

        if mode == "few_shot":
            self.config.freeze_backbone = False
            if int(getattr(self.config, "num_epochs", 0)) <= 0:
                self.config.num_epochs = 2
        elif mode == "zero_shot":
            self.config.num_epochs = 0
            self.config.freeze_backbone = True
        else:
            self.config.freeze_backbone = False

        return mode

    def _sampling_rate(self) -> float:
        raw = float(getattr(self.config, "sampling_rate", getattr(self.config, "few_shot_ratio", 0.1)))
        rate = max(0.0, min(1.0, raw))
        self.config.sampling_rate = rate
        self.config.few_shot_ratio = rate
        return rate

    def _sampling_strategy(self) -> str:
        strategy = str(getattr(self.config, "sampling_strategy", "uniform")).lower()
        if strategy not in {"begin", "end", "uniform", "random"}:
            raise ValueError(f"Unknown sampling_strategy: {strategy}")
        self.config.sampling_strategy = strategy
        return strategy

    # data->先按照原始训练长度剪切，（先缩短原始训练数据，然后再截取窗口）
    # sample-> 先把数据拆成很多个窗口，然后从中收集子集（但是我们一般就是sample）
    def _sampling_basis(self) -> str:
        basis = str(getattr(self.config, "sampling_basis", "data")).lower()
        if basis not in {"data", "sample"}:
            raise ValueError(f"Unknown sampling_basis: {basis}")
        self.config.sampling_basis = basis
        return basis


class TSFMDeepForecastingModelBase(DeepForecastingModelBase):

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        use_dataset_sample = (
            str(getattr(self.config, "shot_mode", "full_shot")).lower() == "few_shot"
        )
        if not use_dataset_sample:
            return super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )

        original_provider = deep_base_module.forecasting_data_provider
        expected_train_call = 1 if float(train_ratio_in_tv) == 1.0 else 2
        provider_call_count = 0

        def _provider_with_few_shot(
            data,
            config,
            timeenc,
            batch_size,
            shuffle,
            drop_last,
            sample_timestamp,
            model_name,
        ):
            nonlocal provider_call_count
            provider_call_count += 1
            dataset, loader = original_provider(
                data,
                config,
                timeenc,
                batch_size,
                shuffle,
                drop_last,
                sample_timestamp,
                model_name,
            )

            if provider_call_count == expected_train_call:
                loader = apply_few_shot_loader(
                    dataset=dataset,
                    loader=loader,
                    config=config,
                    drop_last=drop_last,
                    collate_fn=loader.collate_fn,
                )
            return dataset, loader
        

        # 临时替换函数，将原始的forecasting_data_provider替换为带few-shot功能的版本，确保仅在训练数据加载时应用few-shot策略
        deep_base_module.forecasting_data_provider = _provider_with_few_shot
        try:
            return super().forecast_fit(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
        finally:
            deep_base_module.forecasting_data_provider = original_provider


class BaseTSFMAdapter(TSFMAdapterWorkflowBase, TSFMDeepForecastingModelBase):
    def __init__(self, model_name, model_class, default_hyper_params, **kwargs):
        super().__init__(default_hyper_params, **kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.check_point = None
        self._official_scheduler = None

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        return self.model_class(self.config)

    def _uses_internal_train_loss(self) -> bool:
        return False

    def _init_criterion(self):
        if self._uses_internal_train_loss():
            return _ZeroLoss()
        return super()._init_criterion()

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del exog_future

        if self._uses_internal_train_loss() and hasattr(self.model, "forward_train"):
            pred, train_loss = self.model.forward_train(input, target)
            return {"output": pred, "additional_loss": train_loss}

        dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float().to(input.device)
        output = self.model(input, input_mark, dec_input, target_mark)
        if isinstance(output, tuple):
            output = output[0]
        return {"output": output}

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        mode = self._resolve_shot_mode()
        # 如果是few-shot模式，根据配置设置采样率、采样策略和采样基准
        self._setup_post_norm_shot(mode)

        # if mode == "few_shot":
        #     train_valid_data, covariates = self._apply_few_shot_strategy(
        #         train_valid_data,
        #         covariates,
        #         train_ratio_in_tv,
        #     )

        return super().forecast_fit(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )


def generate_model_factory(
    model_name: str,
    model_class: type,
    adapter_cls: type,
    required_args: dict,
) -> Dict:
    def model_factory(**kwargs):
        return adapter_cls(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


# def tsfm_adapter(model_info: Type[object]) -> object:
#     return moirai_adapter(model_info)
