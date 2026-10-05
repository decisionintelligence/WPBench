from typing import Dict, List, Optional, Type

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    SEMPO_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.tsfm.few_shot_utils import apply_few_shot_loader
from ts_benchmark.baselines.tsfm.timer_helpers import (
    build_timer_ci_loaders,
    ensure_datetime_index,
    fit_wide_external_scaler,
    inverse_transform_wide_2d,
    inverse_transform_wide_3d,
    infer_timer_freq,
    resolve_external_scaler_mode,
    resolve_timer_data_mode,
    transform_wide_2d,
    transform_wide_3d,
)
from ts_benchmark.baselines.utils import forecasting_data_provider, train_val_split
from ts_benchmark.models.model_base import BatchMaker, ModelBase
from ts_benchmark.utils.data_processing import infer_series_number


class SEMPOAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, SEMPO_HPARAMS, **kwargs)

        if hasattr(self.config, "finetune_epochs") and int(getattr(self.config, "finetune_epochs", 0)) > 0:
            self.config.num_epochs = int(self.config.finetune_epochs)

        if not hasattr(self.config, "horizon"):
            self.config.horizon = int(getattr(self.config, "pred_len", 0))
        self.config.pred_len = int(getattr(self.config, "pred_len", self.config.horizon))
        self.config.horizon = int(self.config.pred_len)
        self.config.is_spatial = False
        self.config.timer_data_mode = resolve_timer_data_mode(
            getattr(self.config, "timer_data_mode", "official_ci")
        )
        self.config.external_scaler_mode = resolve_external_scaler_mode(
            getattr(self.config, "external_scaler_mode", "pooled_target")
        )

        if int(getattr(self.config, "label_len", -1)) <= 0:
            self.config.label_len = int(self.config.pred_len)
        else:
            self.config.label_len = int(getattr(self.config, "label_len", 0))

        self.CovariateFusion = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = None
        self.freq = str(getattr(self.config, "freq", "h"))
        self.config.freq = self.freq
        self._timer_total_vars = None
        self._timer_original_series_num = None
        self._timer_original_series_dim = None
        self._train_output_plan: Optional[Dict[str, object]] = None

    @property
    def model_name(self):
        return self._model_name

    def _resolve_shot_mode(self) -> str:
        mode = str(getattr(self.config, "shot_mode", "full_shot")).lower()
        if mode not in {"few_shot", "zero_shot", "full_shot"}:
            raise ValueError(f"Unknown shot_mode: {mode}")

        self.config.shot_mode = mode
        # 确定模式之后，删除对于的变量mode
        self._on_shot_mode_resolved(mode)

        # Match original SEMPO semantics: backbone stays frozen in zero/few/full.
        self.config.freeze_backbone = True
        if mode == "zero_shot":
            self.config.num_epochs = 0
        elif mode == "few_shot" and int(getattr(self.config, "num_epochs", 0)) <= 0:
            self.config.num_epochs = 2

        return mode

    def _init_optimizer(self, CovariateFusion=None):
        del CovariateFusion

        decay_names = set()
        for module_name, module in self.model.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if isinstance(module, nn.LayerNorm):
                    continue
                if "bias" in param_name:
                    continue
                decay_names.add(full_name)

        weight_decay = float(getattr(self.config, "weight_decay", 0.1))
        adam_beta1 = float(getattr(self.config, "adam_beta1", 0.9))
        adam_beta2 = float(getattr(self.config, "adam_beta2", 0.95))

        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if n in decay_names and p.requires_grad
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if n not in decay_names and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            optim_groups,
            lr=float(getattr(self.config, "lr", 1e-4)),
            betas=(adam_beta1, adam_beta2),
            eps=1e-8,
        )

    def _adjust_lr(self, optimizer, epoch, config):
        del optimizer, epoch, config
        # Original SEMPO full-shot defaults effectively keep LR constant in the main train loop.
        return

    def _timer_data_mode(self) -> str:
        mode = resolve_timer_data_mode(getattr(self.config, "timer_data_mode", "official_ci"))
        self.config.timer_data_mode = mode
        return mode

    def _use_official_ci(self) -> bool:
        return self._timer_data_mode() == "official_ci"

    def _base_horizons(self) -> List[int]:
        return [int(h) for h in list(getattr(self.config, "horizon_lengths", [1, 96, 192, 336, 720]))]

    def _build_source_concat_plan(self, horizon: int) -> Dict[str, object]:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError(f"SEMPO horizon must be positive, got {horizon}.")
        # 选择一个所有的horizon集合,来源于sempo自身checkpoint提供的所有预测长度 
        lengths = self._base_horizons()
        greedy_components: List[int] = []
        covered = 0
        # 利用现实存在的length组合，直到目前我们选择的一组组合已经超过pred_len，推出
        while covered < horizon:
            matched = False
            for length in reversed(lengths):
                if covered + length <= horizon:
                    greedy_components.append(length)
                    covered += length
                    matched = True
                    break
            if not matched:
                break

        selected_indices = [i for i, length in enumerate(lengths) if length in greedy_components]
        selected_lengths = [lengths[i] for i in selected_indices]
        concat_total_len = int(sum(selected_lengths))
        # 返回的内容包括，选择的长度对应的预测头的索引，包括选择的长度，最后拼接成的长度，和一组组合（这是一组模型本身支持的预测长度的组合，合并之后的结果对应于pred_len）
        return {
            "mode": "exact",
            "requested_horizon": horizon,
            "selected_indices": selected_indices,
            "selected_lengths": selected_lengths,
            "concat_total_len": concat_total_len,
            "greedy_components": greedy_components,
        }

    def _build_crop_fallback_plan(self, horizon: int) -> Optional[Dict[str, object]]:
        horizon = int(horizon)
        lengths = self._base_horizons()
        candidates = [(idx, length) for idx, length in enumerate(lengths) if length >= horizon]
        if not candidates:
            return None

        selected_idx, selected_length = min(candidates, key=lambda item: item[1])
        # 选出大于目标的最大值
        return {
            "mode": "crop",
            "requested_horizon": horizon,
            "selected_indices": [selected_idx],
            "selected_lengths": [selected_length],
            "concat_total_len": int(selected_length),
            "crop_len": horizon,
        }
    
    # 输入的horizon是pred_len；pred_len是模型每一次吐出的预测长度；
    # 在SEMPO当中，我们都是从长的序列种截取短的
    def _resolve_train_plan(self, horizon: int) -> Dict[str, object]:
        plan = self._build_source_concat_plan(horizon)
        if int(plan["concat_total_len"]) != int(horizon):
            crop_plan = self._build_crop_fallback_plan(horizon)
            extra_guidance = ""
            if crop_plan is not None:
                suggested_pred_len = int(crop_plan["selected_lengths"][0])
                extra_guidance = (
                    " For few/full-shot on a shorter target horizon, keep "
                    f"model-hyper-params.pred_len={suggested_pred_len} for SEMPO head training "
                    f"and set strategy-args.horizon={horizon} with strategy-args.stride={horizon} "
                    "for evaluation."
                )
            raise ValueError(
                "SEMPO training only supports horizons that can be composed exactly from "
                f"{self._base_horizons()} under the original source logic. Got pred_len={horizon}, "
                f"source concat total={plan['concat_total_len']}, greedy components={plan['greedy_components']}."
                f"{extra_guidance}"
            )
        return plan

    def _resolve_infer_plan(self, horizon: int) -> Dict[str, object]:
        # horizon对应了整个任务至少要完成的预测长度
        # exact_plan对应了在模型本身提供的预测长度基础上，选择多个预测头进行拼接（但是这个时候，每个头只能取一次）
        exact_plan = self._build_source_concat_plan(horizon)
        # 本质上，如果预测的长度恰好可以等于任务可以拼接成的最小长度，那么这种代码是合理的，也就会返回所谓的exact plan
        if int(exact_plan["concat_total_len"]) == int(horizon):
            return exact_plan
        # 如果无法适用先用的plan的时候，在test的时候会选择一种现有的组合方式
        crop_plan = self._build_crop_fallback_plan(horizon)
        if crop_plan is not None:
            return crop_plan

        raise ValueError(
            "SEMPO inference cannot serve horizon="
            f"{horizon}. No exact source composition exists and no larger single head is available in "
            f"{self._base_horizons()}."
        )

    def _compose_outputs(self, outputs: List[torch.Tensor], plan: Dict[str, object], series_dim: int) -> torch.Tensor:
        if not outputs:
            raise ValueError("SEMPO returned no prediction heads.")

        channel_dim = min(int(series_dim), int(outputs[0].shape[-1]))
        if channel_dim <= 0:
            raise ValueError(f"Invalid channel_dim={channel_dim} while composing SEMPO outputs.")

        if plan["mode"] == "exact":
            pieces = [
                outputs[idx][:, -int(length) :, :channel_dim]
                for idx, length in zip(plan["selected_indices"], plan["selected_lengths"])
            ]
            if not pieces:
                raise ValueError(f"No SEMPO heads were selected for exact plan {plan}.")
            return torch.cat(pieces, dim=1)
        # 我们仅仅为了适配我们的任务，所以我们是选择一个比他预测长的头截取
        if plan["mode"] == "crop":
            idx = int(plan["selected_indices"][0])
            crop_len = int(plan["crop_len"])
            return outputs[idx][:, :crop_len, :channel_dim]

        raise ValueError(f"Unknown SEMPO compose mode: {plan['mode']}")

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, exog_future
        outputs, recons = self.model(input, input_mark, None, target_mark)
        if recons.shape[1] != input.shape[1]:
            recons = recons[:, -input.shape[1] :, :]
        return {"outputs": outputs, "backcast": recons}

    def _slice_sempo_main_loss(self, output, target, series_dim):
        channel_dim = min(int(series_dim), int(output.shape[-1]), int(target.shape[-1]))
        if output.ndim == 3:
            pred = output[:, -self.config.pred_len :, :channel_dim]
            true = target[:, -self.config.pred_len :, :channel_dim]
            return pred, true

        if output.ndim == 4:
            pred = output[:, :, -self.config.pred_len :, :channel_dim]
            true = target[:, :, -self.config.pred_len :, :channel_dim]
            b, n, t, c = pred.shape
            return pred.reshape(b * n, t, c), true.reshape(b * n, t, c)

        raise ValueError(f"Unexpected output ndim={output.ndim}")

    def _slice_sempo_recons_loss(self, backcast: torch.Tensor, input_tensor: torch.Tensor):
        channel_dim = min(int(backcast.shape[-1]), int(input_tensor.shape[-1]))
        if channel_dim <= 0:
            raise ValueError(
                "SEMPO reconstruction loss received invalid channel dims: "
                f"backcast={tuple(backcast.shape)}, input={tuple(input_tensor.shape)}"
            )
        return backcast[:, :, :channel_dim], input_tensor[:, :, :channel_dim]

    def _validate_timer(self, valid_data_loader, series_dim, criterion):
        total_loss = []
        total_count = []
        self.model.eval()
        plan = self._train_output_plan or self._resolve_train_plan(int(self.config.pred_len))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for i, (input, target, input_mark, target_mark) in enumerate(valid_data_loader):
                batch_size = int(input.shape[0])
                # if i in {0, 1, len(valid_data_loader) - 1}:
                #     print(
                #         f"[BM][VAL] i={i} "
                #         f"input.shape={tuple(input.shape)} "
                #         f"target.shape={tuple(target.shape)} "
                #         f"x00={input[0,0,0].item():.6f} "
                #         f"x_last={input[0,-1,0].item():.6f} "
                #         f"y00={target[0,0,0].item():.6f} "
                #         f"y_last={target[0,-1,0].item():.6f}"
                #     )

                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                output = self._compose_outputs(out_loss["outputs"], plan, series_dim)
                additional_loss = out_loss.get("additional_loss", 0.0)
                backcast = out_loss["backcast"]

                pred_main, true_main = self._slice_sempo_main_loss(output, target, series_dim)
                pred_main, true_main = self._post_process(pred_main, true_main)
                main_loss = criterion(pred_main, true_main)
                recon_pred, recon_true = self._slice_sempo_recons_loss(backcast, input)
                recons_loss = criterion(recon_pred, recon_true)

                if not torch.is_tensor(additional_loss):
                    additional_loss = main_loss.new_tensor(float(additional_loss))

                total_loss.append((main_loss + recons_loss + additional_loss).detach().cpu().item())
                total_count.append(batch_size)

        self.model.train()
        return float(np.average(total_loss)) if total_loss else np.inf

    def _build_timer_loaders_legacy(
        self,
        train_data,
        valid_data,
        sample_train_data,
        sample_valid_data,
        config,
    ):
        train_dataset, train_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=False,
            sample_timestamp=sample_train_data,
            model_name="Timer",
        )

        train_loader = apply_few_shot_loader(
            dataset=train_dataset,
            loader=train_loader,
            config=config,
            drop_last=False,
            collate_fn=train_loader.collate_fn,
        )

        valid_loader = None
        if (
            valid_data is not None
            and sample_valid_data is not None
            and len(valid_data) > (config.seq_len + config.pred_len)
        ):
            _, valid_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=False,
                drop_last=False,
                sample_timestamp=sample_valid_data,
                model_name="Timer",
            )

        return train_loader, valid_loader

    def _configure_official_ci(
        self,
        train_valid_data: pd.DataFrame,
        *,
        series_num: int,
        series_dim: int,
        adj_mx,
        geo_data,
    ) -> None:
        freq = infer_timer_freq(train_valid_data.index, fallback=str(getattr(self.config, "freq", "h")))
        self.config.freq = freq
        self.freq = freq

        self._timer_total_vars = train_valid_data.shape[1]
        self._timer_original_series_num = series_num
        self._timer_original_series_dim = series_dim

        self.config.num_nodes = series_num
        self.config.series_num = series_num
        self.config.series_dim = series_dim
        self.config.timer_total_vars = self._timer_total_vars
        self.config.timer_original_series_num = series_num
        self.config.timer_original_series_dim = series_dim
        self.config.input_dim = 1
        self.config.output_dim = 1
        self.config.enc_in = 1
        self.config.dec_in = 1
        self.config.c_out = 1
        self.config.adj_mx = adj_mx
        self.config.geo_data = geo_data

    def _fit_official_ci_scaler_like_original(
        self,
        train_data: pd.DataFrame,
        valid_data: Optional[pd.DataFrame],
        *,
        norm: bool,
        series_num: int,
        series_dim: int,
    ) -> tuple[StandardScaler, pd.DataFrame, Optional[pd.DataFrame]]:
        mode = resolve_external_scaler_mode(str(getattr(self.config, "external_scaler_mode", "pooled_target")))
        scaler = StandardScaler() if self.scaler1 is None else self.scaler1

        # Match the original SEMPO CI dataset path: keep sklearn fitting in float64
        # and only cast to float32 later when batches become torch tensors.
        train_np = train_data.to_numpy(copy=True)
        if mode == "legacy_wide":
            scaler.fit(train_np)
        else:
            steps = train_np.shape[0]
            pooled = train_np.reshape(steps, int(series_num), int(series_dim)).reshape(
                steps * int(series_num), int(series_dim)
            )
            scaler.fit(pooled)

        if not norm:
            return scaler, train_data, valid_data

        train_scaled_np = transform_wide_2d(
            train_np,
            scaler,
            norm=True,
            mode=mode,
            series_num=series_num,
            series_dim=series_dim,
        )
        train_scaled = pd.DataFrame(train_scaled_np, index=train_data.index, columns=train_data.columns)

        valid_scaled = None
        if valid_data is not None:
            valid_np = valid_data.to_numpy(copy=True)
            valid_scaled_np = transform_wide_2d(
                valid_np,
                scaler,
                norm=True,
                mode=mode,
                series_num=series_num,
                series_dim=series_dim,
            )
            valid_scaled = pd.DataFrame(valid_scaled_np, index=valid_data.index, columns=valid_data.columns)

        return scaler, train_scaled, valid_scaled

    def _prepare_timer_model(self):
        self.model = self._init_model()
        self.CovariateFusion = None
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
        print("----------------------------------------------------------", self.model_name)

    def _prepare_model_for_inference(self, device: torch.device) -> None:
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
        self.model.to(device)
        self.model.eval()

    def _forward_sempo_outputs(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        batch_size, seq_len, _ = input_tensor.shape
        input_mark = torch.zeros(batch_size, seq_len, 1, device=input_tensor.device)
        outputs, _ = self.model(input_tensor, input_mark, None, input_mark)
        return outputs

    def _predict_wide_array(self, scaled_history: np.ndarray, horizon: int) -> np.ndarray:
        if scaled_history.shape[0] < int(self.config.seq_len):
            raise ValueError("History length is shorter than seq_len for SEMPO forecast.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prepare_model_for_inference(device)
        plan = self._resolve_infer_plan(horizon)

        history = np.asarray(scaled_history[-self.config.seq_len :, :], dtype=np.float32)
        input_batch = rearrange(history, "l m -> m l 1")
        input_tensor = torch.tensor(input_batch, dtype=torch.float32, device=device)

        with torch.no_grad():
            outputs = self._forward_sempo_outputs(input_tensor)
            pred = self._compose_outputs(outputs, plan, 1)

        return pred.squeeze(-1).transpose(0, 1).cpu().numpy()

    def _predict_wide_batch(self, scaled_batch: np.ndarray, horizon: int) -> np.ndarray:
        if scaled_batch.shape[1] < int(self.config.seq_len):
            raise ValueError("Batch history length is shorter than seq_len for SEMPO batch_forecast.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 把模型放到卡上，调成测试模式
        self._prepare_model_for_inference(device)
        # 获取推理逻辑
        plan = self._resolve_infer_plan(horizon)

        rolling = np.asarray(scaled_batch[:, -self.config.seq_len :, :], dtype=np.float32)
        total_windows, _, total_vars = rolling.shape
        # 这里把 wide rolling 窗口重排回 CIDatasetBenchmark 的样本顺序：
        # 先取变量 0 的所有窗口，再取变量 1 的所有窗口，依此类推。
        # 这样后面再按 config.batch_size 连续切块时，才能复现原始 CI test loader 的打包语义。
        # 这样子，从开头开始截取，每截取batch_size个窗口都对应了一个完整的变量batch
        ci_stream = rearrange(rolling, "k l m -> (m k) l 1")
        infer_batch_size = max(int(getattr(self.config, "batch_size", 1)), 1)
        ci_predictions = []

        with torch.no_grad():
            for start in range(0, ci_stream.shape[0], infer_batch_size):
                end = min(start + infer_batch_size, ci_stream.shape[0])
                # 每次 forward 送进模型的仍然是普通的 CI 单通道 batch，shape 是 (b, l, 1)。
                # 关键区别在于，这里的 b 来自上面重建后的全局 CI 样本流，
                # 而不是 benchmark 原本 wide-window 顺序下直接切出来的 batch。
                input_tensor = torch.tensor(ci_stream[start:end], dtype=torch.float32, device=device)
                outputs = self._forward_sempo_outputs(input_tensor)
                pred = self._compose_outputs(outputs, plan, 1)
                ci_predictions.append(pred.squeeze(-1).detach().cpu().numpy())

        prediction = np.concatenate(ci_predictions, axis=0)
        return rearrange(prediction, "(m k) t -> k t m", m=total_vars, k=total_windows)

    def _forecast_fit_legacy(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        if covariates is None:
            covariates = {}
        series_num = infer_series_number(train_valid_data)

        self.config.num_nodes = series_num
        series_dim = train_valid_data.shape[-1] // series_num
        if series_num * series_dim != train_valid_data.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        sample_train_valid_data = train_valid_data.iloc[:, :series_dim]
        train_valid_data = self.reshape_spatiotemporal(train_valid_data, series_num)[0]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            exog_dim = exog_data.shape[-1] // series_num
            sample_exog_data = exog_data.iloc[:, :exog_dim]
            exog_data = self.reshape_spatiotemporal(exog_data, series_num)[0]
            train_valid_data = np.concatenate([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-2]
        else:
            exog_dim = 0

        sample_train_valid_data = (
            pd.concat([sample_train_valid_data, sample_exog_data], axis=1)
            if exog_data is not None
            else sample_train_valid_data
        )
        if sample_train_valid_data.shape[1] == 1:
            self.single_forecasting_hyper_param_tune(sample_train_valid_data)
        else:
            self.multi_forecasting_hyper_param_tune(sample_train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim
        self.config.adj_mx = adj_mx
        self.config.series_num = series_num
        self.config.geo_data = kwargs.get("geo_data", None)

        criterion = self._init_criterion()
        self._prepare_timer_model()
        config = self.config
        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        sample_train_data, sample_valid_data = train_val_split(
            sample_train_valid_data, train_ratio_in_tv, config.seq_len
        )
        train_data_len = train_data.shape[0]
        valid_data_len = valid_data.shape[0] if valid_data is not None else 0

        if exog_dim > 0:
            self.scaler1.fit(rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c"))
            self.scaler2.fit(rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c"))
            if config.norm:
                scaled_series = self.scaler1.transform(
                    rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c")
                )
                train_series = rearrange(scaled_series, "(l n) c -> l c n", l=train_data_len)
                scaled_exog = self.scaler2.transform(
                    rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c")
                )
                train_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=train_data_len)
                train_data = np.concatenate((train_series, train_exog), axis=1)
        else:
            self.scaler1.fit(rearrange(train_data, "l c n -> (l n) c"))
            if config.norm:
                scaled_data = self.scaler1.transform(rearrange(train_data, "l c n -> (l n) c"))
                train_data = rearrange(scaled_data, "(l n) c -> l c n", l=train_data_len)

        if train_ratio_in_tv != 1 and config.norm:
            if exog_dim > 0:
                scaled_series = self.scaler1.transform(
                    rearrange(valid_data[:, :series_dim, :], "l c n -> (l n) c")
                )
                valid_series = rearrange(scaled_series, "(l n) c -> l c n", l=valid_data_len)
                scaled_exog = self.scaler2.transform(
                    rearrange(valid_data[:, series_dim:, :], "l c n -> (l n) c")
                )
                valid_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=valid_data_len)
                valid_data = np.concatenate((valid_series, valid_exog), axis=1)
            else:
                scaled_data = self.scaler1.transform(rearrange(valid_data, "l c n -> (l n) c"))
                valid_data = rearrange(scaled_data, "(l n) c -> l c n", l=valid_data_len)

        valid_for_loader = valid_data if train_ratio_in_tv != 1 else None
        sample_valid_for_loader = sample_valid_data if train_ratio_in_tv != 1 else None
        self.train_data_loader, valid_data_loader = self._build_timer_loaders_legacy(
            train_data=train_data,
            valid_data=valid_for_loader,
            sample_train_data=sample_train_data,
            sample_valid_data=sample_valid_for_loader,
            config=config,
        )

        optimizer = self._init_optimizer(CovariateFusion=self.CovariateFusion)
        scaler = torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model total trainable parameters:{total_params}")
        print(f"Total trainable parameters: {total_params}")

        train_plan = None if int(config.num_epochs) <= 0 else (self._train_output_plan or self._resolve_train_plan(int(config.pred_len)))
        for epoch in range(config.num_epochs):
            self.model.train()
            for input, target, input_mark, target_mark in self.train_data_loader:
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                additional_loss = out_loss.get("additional_loss", 0)
                output = self._compose_outputs(out_loss["outputs"], train_plan, series_dim)
                backcast = out_loss["backcast"]

                pred_main, true_main = self._slice_sempo_main_loss(output, target, series_dim)
                output, target = self._post_process(pred_main, true_main)
                recon_pred, recon_true = self._slice_sempo_recons_loss(backcast, input)
                total_loss = criterion(output, target) + criterion(recon_pred, recon_true) + additional_loss

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1:
                valid_loss = self._validate_timer(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint({"Model": self.model})

                if self.early_stopping.early_stop:
                    break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)
        return self

    def _forecast_fit_official_ci(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        del covariates
        # 重构了数据集的时间索引（然后这个结果是不设计到协变量的，我是直接把协变量删除的?但我记得好像timer并没有删除呢？
        train_valid_data = ensure_datetime_index(train_valid_data)
        series_num = infer_series_number(train_valid_data)
        series_dim = train_valid_data.shape[-1] // series_num
        if series_num * series_dim != train_valid_data.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")
        
        #
        self._configure_official_ci(
            train_valid_data,
            series_num=series_num,
            series_dim=series_dim,
            adj_mx=adj_mx,
            geo_data=kwargs.get("geo_data", None),
        )

        criterion = self._init_criterion()
        self._prepare_timer_model()
        config = self.config

        train_data, valid_data = train_val_split(train_valid_data, train_ratio_in_tv, config.seq_len)
        self.scaler1, train_data, valid_data = fit_wide_external_scaler(
            train_data,
            valid_data,
            norm=bool(config.norm),
            scaler=self.scaler1,
            mode=str(config.external_scaler_mode),
            series_num=series_num,
            series_dim=series_dim,
        )
        # 生成了和timer类似的每次都通道独立加载的数据集
        _, self.train_data_loader, valid_data_loader = build_timer_ci_loaders(
            train_data,
            valid_data if train_ratio_in_tv != 1 else None,
            config,
        )
        optimizer = self._init_optimizer(CovariateFusion=None)
        scaler = torch.cuda.amp.GradScaler() if config.use_amp == 1 else None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"model total trainable parameters:{total_params}")
        print(f"Total trainable parameters: {total_params}")

        train_plan = None if int(config.num_epochs) <= 0 else (self._train_output_plan or self._resolve_train_plan(int(config.pred_len)))
        for epoch in range(config.num_epochs):
            self.model.train()
            for batch_idx, (input, target, input_mark, target_mark) in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                input_mark = input_mark.to(device)
                target_mark = target_mark.to(device)

                out_loss = self._process(input, target, input_mark, target_mark, exog_future=None)
                additional_loss = out_loss.get("additional_loss", 0)
                output = self._compose_outputs(out_loss["outputs"], train_plan, 1)
                backcast = out_loss["backcast"]

                # 由于每次预测出来都是单通道，所以channel数量写1即可
                pred_main, true_main = self._slice_sempo_main_loss(output, target, 1)
                output, target = self._post_process(pred_main, true_main)
                recon_pred, recon_true = self._slice_sempo_recons_loss(backcast, input)
                total_loss = criterion(output, target) + criterion(recon_pred, recon_true) + additional_loss
                # 每十轮输出一次epoch
                if (batch_idx) % 100 == 0:
                    print(f"Epoch {batch_idx}, Batch Loss: {total_loss.item():.7f}")

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)

            if train_ratio_in_tv != 1 and valid_data_loader is not None:
                valid_loss = self._validate_timer(valid_data_loader, 1, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    self.check_point = self.save_checkpoint({"Model": self.model})
                if self.early_stopping.early_stop:
                    break

            if self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)
        return self

    def forecast_fit(
        self,
        train_valid_data: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
        train_ratio_in_tv: float = 1.0,
        adj_mx: Optional[dict] = None,
        **kwargs,
    ) -> "ModelBase":
        # 确定模式，存入config后，删除无关的临时变量mode
        mode = self._resolve_shot_mode()
        # 如果mode是few_shot；确定选取few_shot的数据占总体的比例，选取few_shot数据的分布（前后均匀）；
        # 设置选取数据的策略（sample是从整体数据上先划分窗口，然后按比例去选出窗口）data则是先把数据按比例提取，再划分窗口，目前我们只有sample
        self._setup_post_norm_shot(mode)
        if mode == "zero_shot":
            self._train_output_plan = None
        else:
            self._train_output_plan = self._resolve_train_plan(int(self.config.pred_len))
        # 获得train_output_plan计划，用于确定训练的头的组合等等
        if self._use_official_ci():
            return self._forecast_fit_official_ci(
                train_valid_data,
                covariates=covariates,
                train_ratio_in_tv=train_ratio_in_tv,
                adj_mx=adj_mx,
                **kwargs,
            )
        return self._forecast_fit_legacy(
            train_valid_data,
            covariates=covariates,
            train_ratio_in_tv=train_ratio_in_tv,
            adj_mx=adj_mx,
            **kwargs,
        )

    def forecast(
        self,
        horizon: int,
        series: pd.DataFrame,
        *,
        covariates: Optional[dict] = None,
    ) -> np.ndarray:
        if not self._use_official_ci():
            return super().forecast(horizon, series, covariates=covariates)

        del covariates
        wide = ensure_datetime_index(series)
        if self._timer_total_vars is not None and wide.shape[1] != self._timer_total_vars:
            raise ValueError(
                "Forecast input column count does not match the SEMPO training layout: "
                f"expected {self._timer_total_vars}, got {wide.shape[1]}."
            )

        values = transform_wide_2d(
            wide.to_numpy(dtype=np.float32, copy=True),
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._timer_original_series_num),
            series_dim=int(self._timer_original_series_dim),
        )

        pred = self._predict_wide_array(values, horizon)
        return inverse_transform_wide_2d(
            pred,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=int(self._timer_original_series_num),
            series_dim=int(self._timer_original_series_dim),
        )

    def batch_forecast(
        self,
        horizon: int,
        batch_maker: BatchMaker,
        exog_futures,
        i,
        series_number,
        **kwargs,
    ) -> np.ndarray:
        if not self._use_official_ci():
            return super().batch_forecast(horizon, batch_maker, exog_futures, i, series_number, **kwargs)

        del exog_futures, i, kwargs

        if series_number == 1:
            input_batches = []
            if hasattr(batch_maker, "has_more_batches"):
                while batch_maker.has_more_batches():
                    input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
                    input_batches.append(np.asarray(input_data["input"], dtype=np.float32))
            else:
                input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
                input_batches.append(np.asarray(input_data["input"], dtype=np.float32))

            if not input_batches:
                raise RuntimeError("SEMPO batch_forecast received no rolling windows to predict.")

            input_np = np.concatenate(input_batches, axis=0)
            if input_np.shape[-1] % series_number != 0:
                raise ValueError("Data columns cannot be evenly divided by series_num.")

            total_vars = input_np.shape[-1]
            series_dim = total_vars // series_number

            scaled_batch = transform_wide_3d(
                input_np,
                self.scaler1,
                norm=bool(self.config.norm),
                mode=str(self.config.external_scaler_mode),
                series_num=series_number,
                series_dim=series_dim,
            )

            prediction = self._predict_wide_batch(scaled_batch, horizon)
            prediction = inverse_transform_wide_3d(
                prediction,
                self.scaler1,
                norm=bool(self.config.norm),
                mode=str(self.config.external_scaler_mode),
                series_num=series_number,
                series_dim=series_dim,
            )
            return rearrange(prediction, "b t (n c) -> (b n) t c", n=series_number, c=series_dim)

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = np.asarray(input_data["input"], dtype=np.float32)
        if input_np.shape[-1] % series_number != 0:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        batch_size = input_np.shape[0]
        total_vars = input_np.shape[-1]
        series_dim = total_vars // series_number

        scaled_batch = transform_wide_3d(
            input_np,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=series_dim,
        )

        plan = self._resolve_infer_plan(horizon)
        rolling = np.asarray(scaled_batch[:, -self.config.seq_len :, :], dtype=np.float32)
        rolling = rearrange(rolling, "b t (n c) -> b t n c", n=series_number, c=series_dim)
        channel_predictions = []

        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._prepare_model_for_inference(device)
            for channel_index in range(series_dim):
                channel_rolling = rolling[:, :, :, channel_index : channel_index + 1]
                channel_chunks = []
                remaining = int(horizon)

                while remaining > 0:
                    input_batch = rearrange(channel_rolling[:, -self.config.seq_len :, :, 0], "b l n -> (b n) l 1")
                    input_tensor = torch.tensor(input_batch, dtype=torch.float32, device=device)
                    outputs = self._forward_sempo_outputs(input_tensor)
                    pred = self._compose_outputs(outputs, plan, 1)

                    chunk = pred.squeeze(-1).detach().cpu().numpy()
                    chunk = rearrange(chunk, "(b n) t -> b t n", b=batch_size, n=series_number)

                    take = min(remaining, int(chunk.shape[1]))
                    channel_chunks.append(chunk[:, :take, :])
                    channel_rolling = np.concatenate([channel_rolling, chunk[:, :take, :, None]], axis=1)
                    remaining -= take

                channel_prediction = np.concatenate(channel_chunks, axis=1)
                channel_predictions.append(channel_prediction[..., None])

        prediction = np.concatenate(channel_predictions, axis=-1)
        prediction = rearrange(prediction, "b t n c -> b t (n c)")
        prediction = inverse_transform_wide_3d(
            prediction,
            self.scaler1,
            norm=bool(self.config.norm),
            mode=str(self.config.external_scaler_mode),
            series_num=series_number,
            series_dim=series_dim,
        )
        return rearrange(prediction, "b t (n c) -> (b n) t c", n=series_number, c=series_dim)


def sempo_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=SEMPOAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "pred_len": "output_chunk_length",
        },
    )


def tsfm_adapter(model_info: Type[object]) -> object:
    return sempo_adapter(model_info)
