from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn

_LOCAL_TIMESFM_2P0_DIR = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "timesfm-2.0-500m-pytorch"
)
_LOCAL_TIMESFM_CKPT = _LOCAL_TIMESFM_2P0_DIR / "torch_model.ckpt"

from ts_benchmark.baselines.tsfm.submodules.timesfm import (
    TimesFm,
    TimesFmCheckpoint,
    TimesFmHparams,
)
from ts_benchmark.baselines.tsfm.submodules.timesfm.timesfm_base import DEFAULT_QUANTILES
from ts_benchmark.baselines.tsfm.submodules.timesfm.layers.pytorch_patched_decoder import (
    PatchedTimeSeriesDecoder,
    TimesFMConfig,
)
from ts_benchmark.baselines.tsfm.timesfm_helpers import preprocess_timesfm_inputs


class TimesFM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.shot_mode = str(getattr(configs, "shot_mode", "full_shot")).lower()
        self.backend = str(getattr(configs, "backend", "cpu"))
        self.repo_id = str(
            getattr(configs, "timesfm_repo_id", "google/timesfm-2.0-500m-pytorch")
        )
        raw_ckpt_path = str(getattr(configs, "timesfm_checkpoint_path", "")).strip()
        self.ckpt_path = self._resolve_checkpoint_path(raw_ckpt_path)
        self.use_positional_embedding = bool(
            getattr(configs, "timesfm_use_positional_embedding", False)
        )

        self.infer_model = None
        self.finetune_model = None

        if self.shot_mode == "zero_shot":
            self.infer_model = self._build_infer_model()
        else:
            self.finetune_model = self._build_finetune_model(load_pretrained=True)

    def _resolve_checkpoint_path(self, raw_path: str) -> str:
        if raw_path:
            candidate = Path(raw_path).expanduser()
            if candidate.is_dir():
                candidate = candidate / "torch_model.ckpt"
            return str(candidate)
        if _LOCAL_TIMESFM_CKPT.exists():
            return str(_LOCAL_TIMESFM_CKPT)
        return ""

    def _build_hparams(self):
        return TimesFmHparams(
            backend=self.backend,
            per_core_batch_size=int(self.configs.timesfm_per_core_batch_size),
            context_len=int(self.configs.timesfm_context_len),
            horizon_len=int(self.configs.timesfm_horizon_len),
            input_patch_len=int(self.configs.timesfm_input_patch_len),
            output_patch_len=int(self.configs.timesfm_output_patch_len),
            num_layers=int(self.configs.timesfm_num_layers),
            num_heads=int(self.configs.timesfm_num_heads),
            model_dims=int(self.configs.timesfm_model_dims),
            point_forecast_mode=str(self.configs.timesfm_point_forecast_mode),
            use_positional_embedding=self.use_positional_embedding,
        )

    def _build_infer_model(self):
        ckpt = TimesFmCheckpoint(
            path=self.ckpt_path or None,
            huggingface_repo_id=None if self.ckpt_path else self.repo_id,
        )
        return TimesFm(hparams=self._build_hparams(), checkpoint=ckpt)

    def _build_finetune_model(self, load_pretrained=True):
        # Keep the decoder head aligned with the pretrained checkpoint while the
        # benchmark horizon controls how many forecast steps we train/evaluate.
        arch_horizon = int(getattr(self.configs, "timesfm_horizon_len", 128))
        model = PatchedTimeSeriesDecoder(
            TimesFMConfig(
                num_layers=int(self.configs.timesfm_num_layers),
                num_heads=int(self.configs.timesfm_num_heads),
                hidden_size=int(self.configs.timesfm_model_dims),
                intermediate_size=int(self.configs.timesfm_model_dims),
                patch_len=int(self.configs.timesfm_input_patch_len),
                horizon_len=arch_horizon,
                use_positional_embedding=self.use_positional_embedding,
            )
        )
        if load_pretrained:
            ckpt_path = self.ckpt_path or str(
                Path(snapshot_download(self.repo_id)) / "torch_model.ckpt"
            )
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
        return model

    def zero_shot_forecast(self, inputs, freq):
        return self.infer_model.forecast(inputs, freq=freq)

    def _select_finetune_point_forecast(self, mean_output, full_output):
        mode = str(
            getattr(self.configs, "timesfm_point_forecast_mode", "median")
        ).lower()
        if mode == "mean":
            return mean_output
        if mode == "median":
            quantiles = tuple(getattr(self.configs, "timesfm_quantiles", DEFAULT_QUANTILES))
            try:
                median_index = quantiles.index(0.5)
            except ValueError as exc:
                raise ValueError(
                    f"Median (0.5) is not found in quantiles: {quantiles}."
                ) from exc
            return full_output[:, :, 1 + median_index]
        raise ValueError(
            f"Unsupported point forecast mode: {mode}. Use 'mean' or 'median'."
        )

    def finetune_forecast(self, inputs, freq):
        """Rolling / batch inference on the patched decoder after finetuning."""
        if self.finetune_model is None:
            raise ValueError(
                "finetune_model is not initialized (use non-zero_shot shot_mode)."
            )
        hz = int(self.configs.horizon)
        ctx = int(self.configs.timesfm_context_len)
        bs = max(1, int(self.configs.timesfm_per_core_batch_size))
        input_ts, input_padding, inp_freq, pmap_pad = preprocess_timesfm_inputs(
            inputs,
            freq,
            context_len=ctx,
            horizon_len=hz,
            global_batch_size=bs,
        )
        model = self.finetune_model
        model.eval()
        device = next(model.parameters()).device
        point_chunks = []
        with torch.no_grad():
            n = input_ts.shape[0]
            for i in range(0, n, bs):
                t_input_ts = torch.tensor(
                    input_ts[i : i + bs], dtype=torch.float32, device=device
                )
                t_input_padding = torch.tensor(
                    input_padding[i : i + bs], dtype=torch.float32, device=device
                )
                t_inp_freq = torch.tensor(
                    inp_freq[i : i + bs], dtype=torch.long, device=device
                )
                mean_output, full_output = model.decode(
                    input_ts=t_input_ts,
                    paddings=t_input_padding,
                    freq=t_inp_freq,
                    horizon_len=hz,
                    return_forecast_on_context=False,
                )
                point_output = self._select_finetune_point_forecast(
                    mean_output, full_output
                )
                if point_output.is_cuda:
                    point_output = point_output.cpu()
                point_chunks.append(point_output.detach().numpy())
        point_outputs = np.concatenate(point_chunks, axis=0)
        if pmap_pad > 0:
            point_outputs = point_outputs[:-pmap_pad, ...]
        return point_outputs, None

    def forward(self, x_context, x_padding, freq):
        return self.finetune_model(x_context, x_padding.float(), freq)
