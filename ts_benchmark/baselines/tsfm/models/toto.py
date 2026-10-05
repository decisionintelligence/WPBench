import json
import os
import re
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors.torch as safetorch
from huggingface_hub import ModelHubMixin, constants, hf_hub_download
from torch import nn

from ts_benchmark.baselines.tsfm.submodules.toto.layers.attention import XFORMERS_AVAILABLE
from ts_benchmark.baselines.tsfm.submodules.toto.layers.transformer import XFORMERS_SWIGLU_AVAILABLE
from ts_benchmark.baselines.tsfm.submodules.toto.models.backbone import TotoBackbone


class Toto(nn.Module, ModelHubMixin):
    def __init__(self, configs=None, **kwargs):
        super().__init__()

        def _get(name, default):
            if name in kwargs:
                return kwargs[name]
            if configs is not None:
                if isinstance(configs, dict):
                    return configs.get(name, default)
                return getattr(configs, name, default)
            return default

        patch_size = int(_get("patch_size", 64))
        stride = int(_get("stride", 64))
        embed_dim = int(_get("embed_dim", 768))
        num_layers = int(_get("num_layers", 12))
        num_heads = int(_get("num_heads", 12))
        mlp_hidden_dim = int(_get("mlp_hidden_dim", 3072))
        dropout = float(_get("dropout", 0.1))
        spacewise_every_n_layers = int(_get("spacewise_every_n_layers", 12))
        scaler_cls = _get(
            "scaler_cls",
            "<class 'model.scaler.CausalPatchStdMeanScaler'>",
        )
        output_distribution_classes = _get("output_distribution_classes", None)
        if output_distribution_classes is None:
            output_distribution_classes = [
                "<class 'model.distribution.MixtureOfStudentTsOutput'>",
            ]
        spacewise_first = bool(_get("spacewise_first", False))
        output_distribution_kwargs = _get("output_distribution_kwargs", None)
        if output_distribution_kwargs is None:
            output_distribution_kwargs = {"k_components": 1}
        use_memory_efficient_attention = bool(
            _get("use_memory_efficient_attention", False)
        )
        stabilize_with_global = bool(_get("stabilize_with_global", True))
        scale_factor_exponent = float(_get("scale_factor_exponent", 10.0))

        _backbone_keys = {
            "patch_size",
            "stride",
            "embed_dim",
            "num_layers",
            "num_heads",
            "mlp_hidden_dim",
            "dropout",
            "spacewise_every_n_layers",
            "scaler_cls",
            "output_distribution_classes",
            "spacewise_first",
            "output_distribution_kwargs",
            "use_memory_efficient_attention",
            "stabilize_with_global",
            "scale_factor_exponent",
        }
        self.model_kwargs = {k: v for k, v in kwargs.items() if k not in _backbone_keys}

        self.model = TotoBackbone(
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            dropout=dropout,
            spacewise_every_n_layers=spacewise_every_n_layers,
            scaler_cls=scaler_cls,
            output_distribution_classes=output_distribution_classes,
            spacewise_first=spacewise_first,
            output_distribution_kwargs=output_distribution_kwargs,
            use_memory_efficient_attention=use_memory_efficient_attention,
            stabilize_with_global=stabilize_with_global,
            scale_factor_exponent=scale_factor_exponent,
        )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path,
        map_location: str = "cpu",
        strict=True,
        **model_kwargs,
    ):
        """
        Custom checkpoint loading. Used to load a local
        safetensors checkpoint with an optional config.json file.
        """
        if os.path.isdir(checkpoint_path):
            safetensors_file = os.path.join(checkpoint_path, "model.safetensors")
        else:
            safetensors_file = checkpoint_path

        if os.path.exists(safetensors_file):
            model_state = safetorch.load_file(safetensors_file, device=map_location)
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {safetensors_file}")

        # Load configuration from config.json if it exists.
        config_file = os.path.join(checkpoint_path, "config.json")
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)

        # Merge any extra kwargs into the configuration.
        config.update(model_kwargs)

        remapped_state_dict = cls._map_state_dict_keys(
            model_state, XFORMERS_SWIGLU_AVAILABLE and not config.get("pre_xformers_checkpoint", False)
        )

        if not XFORMERS_AVAILABLE and config.get("use_memory_efficient_attention", True):
            config["use_memory_efficient_attention"] = False

        instance = cls(**config)
        instance.to(map_location)

        # Filter out unexpected keys
        filtered_remapped_state_dict = {
            k: v
            for k, v in remapped_state_dict.items()
            if k in instance.state_dict() and not k.endswith("rotary_emb.freqs")
        }

        instance.load_state_dict(filtered_remapped_state_dict, strict=strict)
        return instance

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=constants.SAFETENSORS_SINGLE_FILE,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            return cls.load_from_checkpoint(model_file, map_location, strict, **model_kwargs)

    @staticmethod
    def _map_state_dict_keys(state_dict, use_fused_swiglu):
        """
        Maps the keys of a state_dict to match the current model's state_dict.
        Currently this is only used to convert between fused and unfused SwiGLU implementations.
        """
        if use_fused_swiglu:
            remap_keys = {
                "mlp.0.weight": "mlp.0.w12.weight",
                "mlp.0.bias": "mlp.0.w12.bias",
                "mlp.2.weight": "mlp.0.w3.weight",
                "mlp.2.bias": "mlp.0.w3.bias",
            }
        else:
            remap_keys = {
                "mlp.0.w12.weight": "mlp.0.weight",
                "mlp.0.w12.bias": "mlp.0.bias",
                "mlp.0.w3.weight": "mlp.2.weight",
                "mlp.0.w3.bias": "mlp.2.bias",
            }

        def replace_key(text):
            for pattern, replacement in remap_keys.items():
                text = re.sub(pattern, replacement, text)
            return text

        return {replace_key(k): v for k, v in state_dict.items()}

    @property
    def device(self):
        return next(self.model.parameters()).device