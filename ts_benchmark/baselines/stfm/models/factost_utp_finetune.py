"""Thin benchmark wrapper around the vendored FactoST UTP fine-tune model."""

from torch import nn

from ts_benchmark.baselines.stfm.submodules.factost.models.FactoST_finetune import (
    FactoST as VendoredFactoSTUTPFinetune,
)


def _get_config_value(configs, *names, default=None):
    for name in names:
        if hasattr(configs, name):
            return getattr(configs, name)
    return default


class FactoST_UTP_Finetune(nn.Module):
    """Benchmark wrapper for upstream FactoST UTP fine-tuning semantics."""

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        seq_len = int(
            _get_config_value(configs, "seq_len", "context_points", default=512)
        )
        horizon = int(
            _get_config_value(configs, "horizon", "pred_len", "target_points", default=96)
        )
        patch_len = int(_get_config_value(configs, "patch_len", default=16))
        num_nodes = int(_get_config_value(configs, "num_nodes", "c_in", default=1))
        num_token = int(
            _get_config_value(configs, "factost_num_token", "num_token", default=3)
        )
        n_layers = int(_get_config_value(configs, "factost_n_layers", "n_layers", default=3))
        n_heads = int(_get_config_value(configs, "factost_n_heads", "n_heads", default=4))
        d_model = int(_get_config_value(configs, "factost_d_model", "d_model", default=256))
        d_ff = int(_get_config_value(configs, "factost_d_ff", "d_ff", default=1024))
        attn_dropout = float(
            _get_config_value(configs, "factost_dropout", "dropout", default=0.2)
        )

        configs.seq_len = seq_len
        configs.context_points = seq_len
        configs.horizon = horizon
        configs.pred_len = horizon
        configs.target_points = horizon
        configs.patch_len = patch_len
        configs.num_nodes = num_nodes

        self.model = VendoredFactoSTUTPFinetune(
            target_dim=horizon,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            num_token=num_token,
            attn_dropout=attn_dropout,
        )

        ckpt_path = str(
            _get_config_value(
                configs,
                "pretrain_model_path",
                "ckpt_path",
                "pretrained_model",
                "factost_checkpoint_path",
                default="",
            )
            or ""
        ).strip()
        if ckpt_path:
            self.model.load_backbone_weights(ckpt_path)

    def load_backbone_weights(self, path: str) -> None:
        self.model.load_backbone_weights(path)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
        **kwargs,
    ):
        del x_mark_enc, x_dec, x_mark_dec, kwargs

        if x_enc.ndim == 4 and x_enc.shape[-1] == 1:
            x_enc = x_enc.squeeze(-1)
        if x_enc.ndim != 3:
            raise ValueError(
                "FactoST_UTP_Finetune expects x_enc with shape [B, T, N], "
                f"got {tuple(x_enc.shape)}."
            )

        return self.model(x_enc, mask=mask)
