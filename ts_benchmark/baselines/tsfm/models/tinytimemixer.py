import torch
from torch import nn

from ts_benchmark.baselines.tsfm.submodules.tinytimemixer.modeling_tinytimemixer import (
    TinyTimeMixerForPrediction,
)
from ts_benchmark.baselines.tsfm.tinytimemixer_helpers import (
    get_ttm_frequency_token,
    resolve_ttm_revision,
)


class TinyTimeMixer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.context_length = int(getattr(configs, "seq_len", 512))
        self.ttm_context_length = int(
            getattr(configs, "ttm_context_length", self.context_length)
        )
        self.prediction_length = int(
            getattr(configs, "pred_len", getattr(configs, "horizon", 96))
        )
        self.freq = str(getattr(configs, "freq", "h"))
        self.head_dropout = float(getattr(configs, "head_dropout", 0.2))
        self.use_frequency_token = bool(getattr(configs, "use_frequency_token", True))
        self.num_input_channels = int(
            getattr(configs, "input_dim", getattr(configs, "num_input_channels", 1))
        )
        self.prediction_channel_indices = getattr(configs, "prediction_channel_indices", None)
        self.exogenous_channel_indices = getattr(configs, "exogenous_channel_indices", None)

        self.ttm_model_card = str(
            getattr(configs, "ttm_model_card", "ibm-granite/granite-timeseries-ttm-r2")
        )
        self.ttm_yaml_path = str(getattr(configs, "ttm_yaml_path", "")) or None
        self.ttm_checkpoint_path = str(getattr(configs, "ttm_checkpoint_path", "")).strip() or None

        revision, native_pred_len, release = resolve_ttm_revision(
            context_length=self.ttm_context_length,
            prediction_length=self.prediction_length,
            model_card=self.ttm_model_card,
            yaml_path=self.ttm_yaml_path,
        )
        self.native_pred_len = native_pred_len
        self.revision = revision
        self.release = release

        self.freq_token_id = get_ttm_frequency_token(self.freq)

        self.model = TinyTimeMixerForPrediction.from_pretrained(
            self.ttm_checkpoint_path or self.ttm_model_card,
            revision=None if self.ttm_checkpoint_path else self.revision,
            head_dropout=self.head_dropout,
            num_input_channels=self.num_input_channels,
            prediction_channel_indices=self.prediction_channel_indices,
            exogenous_channel_indices=self.exogenous_channel_indices,
        )

        if bool(getattr(configs, "freeze_backbone", False)):
            for p in self.model.backbone.parameters():
                p.requires_grad = False

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        del x_mark_enc, x_dec, x_mark_dec, mask

        if x_enc.shape[1] < self.ttm_context_length:
            raise ValueError(
                f"TTM requires input length >= {self.ttm_context_length}, got {x_enc.shape[1]}"
            )
        if x_enc.shape[1] > self.ttm_context_length:
            x_enc = x_enc[:, -self.ttm_context_length :, :]

        model_kwargs = {"past_values": x_enc}
        if self.use_frequency_token:
            model_kwargs["freq_token"] = torch.full(
                (x_enc.shape[0],),
                self.freq_token_id,
                dtype=torch.long,
                device=x_enc.device,
            )
        outputs = self.model(**model_kwargs)
        return outputs.prediction_outputs
