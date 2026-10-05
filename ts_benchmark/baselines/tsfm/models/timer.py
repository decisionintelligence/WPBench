from torch import nn

from ts_benchmark.baselines.tsfm.submodules.timer.models.Timer import TimerModel
from einops import rearrange, repeat


class Timer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        if not hasattr(configs, "task_name"):
            configs.task_name = "forecast"

        ckpt_path = getattr(configs, "ckpt_path", "")
        pretrain_model_path = getattr(configs, "pretrain_model_path", "")
        if not ckpt_path and pretrain_model_path:
            configs.ckpt_path = pretrain_model_path

        if not hasattr(configs, "output_attention"):
            configs.output_attention = False

        self.model = TimerModel(configs)

        if bool(getattr(configs, "freeze_backbone", False)):
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, L, C]
        b, _, c = x_enc.shape

        x_enc_ci = rearrange(x_enc, "b l c -> (b c) l 1")
        x_dec_ci = rearrange(x_dec, "b l c -> (b c) l 1") if x_dec is not None else None

        x_mark_enc_ci = repeat(x_mark_enc, "b l f -> (b c) l f", c=c) if x_mark_enc is not None else None
        x_mark_dec_ci = repeat(x_mark_dec, "b l f -> (b c) l f", c=c) if x_mark_dec is not None else None

        output = self.model(
            x_enc_ci,
            x_mark_enc_ci,
            x_dec_ci,
            x_mark_dec_ci,
            mask=mask,
        )
        if isinstance(output, tuple):
            output = output[0]

        # [(B*C), Lout, 1] -> [B, Lout, C]
        output = rearrange(output, "(b c) l 1 -> b l c", b=b, c=c)
        return output
