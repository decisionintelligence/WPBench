from typing import Optional, Type

import pandas as pd
from torch.utils.data import DataLoader
import numpy as np

import ts_benchmark.baselines.deep_forecasting_model_base as deep_base_module
from ts_benchmark.baselines.stfm.adapters_for_stfm import (
    BaseSTFMAdapter,
    GENERIC_STFM_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.stfm.models.opencity_io import (
    OPENCITY_CHECKPOINT,
    OPENCITY_CONTEXT,
    OPENCITY_HORIZON,
    build_opencity_inputs,
    build_opencity_time_slots,
    restore_opencity_output,
)
from ts_benchmark.baselines.utils import DatasetForTransformer

OPENCITY_HPARAMS = {
    **GENERIC_STFM_HPARAMS,
    "batch_size": 2,
    "is_spatial": True,
    "label_len": 0,
    "norm": True,
    "loss": "MAE",
    "shot_mode": "zero_shot",
    "pretrain_model_path": OPENCITY_CHECKPOINT,
    # OpenCity-plus METR_LA checkpoint profile. Keep native 288->288 geometry
    # and slice benchmark horizons after inference.
    "opencity_internal_context": OPENCITY_CONTEXT,
    "opencity_internal_horizon": OPENCITY_HORIZON,
    "opencity_embed_dim": 512,
    "opencity_skip_dim": 512,
    "opencity_enc_depth": 6,
    "opencity_dataset": "METR_LA",
    "opencity_loss": "l1",
    "opencity_loss_mask_value": None,
    "prefix_horizon_slice": True,
    "freeze_backbone": True,
}


class OpenCityAdapter(BaseSTFMAdapter):
    """Benchmark adapter for OpenCity checkpoint zero-shot and finetuning."""

    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, OPENCITY_HPARAMS, **kwargs)
        self._configure_internal_geometry()

    def _configure_internal_geometry(self) -> None:
        if not hasattr(self.config, "benchmark_horizon"):
            self.config.benchmark_horizon = int(getattr(self.config, "horizon", OPENCITY_HORIZON))
        self.config.horizon = int(getattr(self.config, "opencity_internal_horizon", OPENCITY_HORIZON))
        self.config.pred_len = self.config.horizon
        self.config.seq_len = int(getattr(self.config, "seq_len", OPENCITY_CONTEXT * 2))
        self.config.label_len = 0
        self.config.is_spatial = True
        # self.config.norm = True
        self.config.prefix_horizon_slice = True

    def _resolve_shot_mode(self) -> str:
        mode = str(getattr(self.config, "shot_mode", "zero_shot")).lower()
        if mode not in {"zero_shot", "few_shot", "full_shot", "ori"}:
            raise ValueError(f"Unknown OpenCity shot_mode: {mode}")

        self.config.shot_mode = mode
        if mode == "zero_shot":
            self.config.num_epochs = 0
            self.config.freeze_backbone = True
        elif mode in {"few_shot", "full_shot"}:
            if int(getattr(self.config, "num_epochs", 0)) <= 0:
                self.config.num_epochs = 1
            self.config.freeze_backbone = bool(getattr(self.config, "freeze_backbone", True))
        else:
            self.config.freeze_backbone = False
        return mode

    def _setup_post_norm_shot(self, mode: str) -> None:
        if mode == "few_shot":
            self._sampling_rate()
            self._sampling_strategy()
            self.config.sampling_basis = "sample"
            self._sampling_basis()
            return
        if mode in {"zero_shot", "full_shot", "ori"}:
            return
        raise ValueError(f"Unknown OpenCity shot_mode: {mode}")

    def _uses_internal_train_loss(self) -> bool:
        return True

    @staticmethod
    def _mae_torch(pred, true, mask_value=None):
        if mask_value is not None:
            mask = true > mask_value
            pred = pred.masked_select(mask)
            true = true.masked_select(mask)
            true_count = mask.sum().item()
        else:
            true_count = None

        if true.numel() == 0:
            return pred.new_zeros(()), true_count

        mae_loss = (true - pred).abs()
        return mae_loss.mean(), true_count

    def _opencity_train_loss(self, pred, true):
        loss_name = str(getattr(self.config, "opencity_loss", "l1")).strip().lower()
        if loss_name in {"l1", "mae", "plain_l1", "plain_mae"}:
            return self._mae_torch(pred, true, mask_value=None)[0]
        if loss_name in {"native_mask_mae", "mask_mae"}:
            mask_value = getattr(self.config, "opencity_loss_mask_value", 0.0)
            return self._mae_torch(pred, true, mask_value=mask_value)[0]
        raise ValueError(f"Unknown OpenCity loss: {loss_name}")

    @staticmethod
    def _opencity_data_provider(
        data,
        config,
        timeenc,
        batch_size,
        shuffle,
        drop_last,
        sample_timestamp,
        model_name,
    ):
        del model_name
        dataset = DatasetForTransformer(
            dataset=data,
            sample_timestamp=sample_timestamp,
            history_len=config.seq_len,
            prediction_len=config.pred_len,
            label_len=config.label_len,
            timeenc=timeenc,
            freq=config.freq,
            model_name="OpenCity_STFM",
        )
        raw_dates = pd.to_datetime(sample_timestamp.reset_index()["date"])
        dataset.data_stamp = build_opencity_time_slots(raw_dates)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            drop_last=drop_last,
            collate_fn=None,
        )
        return dataset, loader

    @staticmethod
    def _build_padding_timestamps(time_stamps_list, padding_len):
        padded = []
        for time_stamps in time_stamps_list:
            dates = pd.to_datetime(time_stamps)
            if len(dates) > 1:
                freq = pd.infer_freq(dates)
                if freq is None:
                    freq = dates[-1] - dates[-2]
            else:
                freq = "5min"
            expand_time_stamp = pd.date_range(
                start=dates[-1],
                periods=padding_len + 1,
                freq=freq,
            )
            padded.append(expand_time_stamp.to_numpy()[-padding_len:])
        return padded

    def _padding_time_stamp_mark(self, time_stamps_list, padding_len):
        padded = self._build_padding_timestamps(time_stamps_list, padding_len)
        whole_time_stamp = []
        for original, extra in zip(time_stamps_list, padded):
            whole_time_stamp.append(pd.to_datetime(list(original) + list(extra)).to_numpy())
        return build_opencity_time_slots(np.stack(whole_time_stamp))

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx=None,
        **kwargs,
    ):
        self._configure_internal_geometry()
        original_provider = deep_base_module.forecasting_data_provider
        deep_base_module.forecasting_data_provider = self._opencity_data_provider
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

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del exog_future
        x_open, y_open, y_value = build_opencity_inputs(
            input,
            target,
            input_mark,
            target_mark,
            context_len=int(getattr(self.config, "opencity_internal_context", OPENCITY_CONTEXT)),
            output_len=int(getattr(self.config, "opencity_internal_horizon", OPENCITY_HORIZON)),
        )
        raw_output = self.model(x_open, y_open)
        output = restore_opencity_output(raw_output)

        train_loss = raw_output.new_zeros(())
        if y_value is not None and y_value.shape == raw_output.shape:
            train_loss = self._opencity_train_loss(raw_output, y_value)

        # if bool(getattr(self.config, "opencity_print_shapes", False)):
        #     print(
        #         "OpenCity shapes: "
        #         f"benchmark_input={tuple(input.shape)}, "
        #         f"internal_input={tuple(x_open.shape)}, "
        #         f"raw_output={tuple(raw_output.shape)}, "
        #         f"benchmark_output={tuple(output.shape)}"
        #     )

        return {"output": output, "additional_loss": train_loss}


def opencity_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=OpenCityAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )
