"""Thin benchmark wrapper around the vendored FactoST STA model.

The no-holiday v1 benchmark adapter supplies inputs with shape [B, T, N, 6]:
one value channel plus FactoST's five fixed timestamp slots. This wrapper only
validates the rank-4 model contract; it does not generate or shrink timestamp
channels.
"""

from torch import nn

from ts_benchmark.baselines.stfm.models.factost_time_features import (
    parse_active_temporal_features,
)
from ts_benchmark.baselines.stfm.submodules.factost.models.FactoST import (
    FactoST as VendoredFactoST,
)


def _get_config_value(configs, *names, default=None):
    for name in names:
        if hasattr(configs, name):
            return getattr(configs, name)
    return default


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(value)


def _parse_filter_matrices(spec) -> list[str]:
    if spec is None:
        return ["S_s", "S_t", "S_d"]
    if isinstance(spec, (list, tuple)):
        return [str(item).strip() for item in spec if str(item).strip()]

    text = str(spec).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


class FactoST_STA(nn.Module):
    """Benchmark wrapper for the vendored FactoST STA model."""

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
        stride = int(_get_config_value(configs, "stride", default=patch_len))
        num_nodes = int(_get_config_value(configs, "num_nodes", "c_in", default=1))
        num_token = int(_get_config_value(configs, "factost_num_token", "num_token", default=3))
        embedding_dim = int(
            _get_config_value(configs, "factost_embedding_dim", "embedding_dim", default=32)
        )
        max_delay_steps = int(
            _get_config_value(
                configs,
                "factost_max_delay_steps",
                "max_delay_steps",
                default=3,
            )
        )
        n_layers = int(_get_config_value(configs, "factost_n_layers", "n_layers", default=3))
        n_heads = int(_get_config_value(configs, "factost_n_heads", "n_heads", default=4))
        d_model = int(_get_config_value(configs, "factost_d_model", "d_model", default=256))
        d_ff = int(_get_config_value(configs, "factost_d_ff", "d_ff", default=1024))
        attn_dropout = float(
            _get_config_value(configs, "factost_dropout", "dropout", default=0.2)
        )

        factost_temporal_spec = _get_config_value(
            configs, "factost_temporal_features", default=None
        )
        temporal_spec = (
            factost_temporal_spec
            if factost_temporal_spec is not None
            else _get_config_value(configs, "temporal_features", default=None)
        )
        temporal_features = parse_active_temporal_features(temporal_spec)

        use_st_metadata = _as_bool(
            _get_config_value(
                configs,
                "factost_use_st_metadata",
                "use_st_metadata",
                default=bool(temporal_features),
            )
        )
        use_cpr = _as_bool(
            _get_config_value(
                configs,
                "factost_use_cpr",
                "use_cpr",
                default=use_st_metadata and bool(temporal_features),
            )
        )
        use_st_filtering = _as_bool(
            _get_config_value(
                configs,
                "factost_use_st_filtering",
                "use_st_filtering",
                default=use_st_metadata,
            )
        )
        filter_matrices = _parse_filter_matrices(
            _get_config_value(
                configs,
                "factost_filter_matrices",
                "filter_matrices",
                default="S_s,S_t,S_d",
            )
        )

        configs.seq_len = seq_len
        configs.context_points = seq_len
        configs.horizon = horizon
        configs.pred_len = horizon
        configs.target_points = horizon
        configs.patch_len = patch_len
        configs.stride = stride
        configs.num_nodes = num_nodes
        configs.temporal_features = temporal_features

        self.model = VendoredFactoST(
            target_dim=horizon,
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            num_token=num_token,
            num_nodes=num_nodes,
            temporal_features=temporal_features,
            embedding_dim=embedding_dim,
            use_st_filtering=use_st_filtering,
            filter_matrices=filter_matrices,
            max_delay_steps=max_delay_steps,
            use_st_metadata=use_st_metadata,
            use_cpr=use_cpr,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
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
        future_temporal_features=None,
        mask=None,
        **kwargs,
    ):
        del x_mark_enc, x_dec

        if x_enc.ndim == 3:
            x_enc = x_enc.unsqueeze(-1)
        if x_enc.ndim != 4:
            raise ValueError(
                f"FactoST_STA expects x_enc with shape [B, T, N, C], got {tuple(x_enc.shape)}."
            )

        if future_temporal_features is None:
            future_temporal_features = x_mark_dec

        return self.model(
            x_enc,
            future_temporal_features=future_temporal_features,
            mask=mask,
            **kwargs,
        )
