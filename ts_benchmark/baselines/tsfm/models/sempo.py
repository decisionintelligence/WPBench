import torch
from torch import nn
from ts_benchmark.baselines.tsfm.submodules.sempo.models.SEMPO import SEMPOModel


class SEMPO(nn.Module):
    def __init__(self, configs):
        super().__init__()
        configs.head_type = "prediction"
        configs.c_in = 1
        configs.pred_len = int(getattr(configs, "pred_len", getattr(configs, "horizon", 96)))
        configs.horizon = configs.pred_len
        configs.horizon_lengths = list(getattr(configs, "horizon_lengths", [1, 96, 192, 336, 720]))
        self.model = SEMPOModel(configs)

        ckpt = getattr(configs, "pretrain_model_path", "") or getattr(configs, "ckpt_path", "")
        if ckpt:
            self._load_ckpt(ckpt)

        if bool(getattr(configs, "freeze_backbone", True)):
            self.model.freeze_backbone()

    def _load_ckpt(self, path: str):
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {}
        for k, v in state.items():
            kk = k[7:] if k.startswith("module.") else k
            cleaned[kk] = v
        self.model.load_state_dict(cleaned, strict=False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
