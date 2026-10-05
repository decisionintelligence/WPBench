from typing import Type

from ts_benchmark.baselines.tsfm.adapters_for_tsfm import (
    BaseTSFMAdapter,
    TIME_LLM_HPARAMS,
    generate_model_factory,
)


def _infer_llm_dim(config, llm_dim):
    model_name = str(getattr(config, "llm_model", "")).upper()
    model_id_map = {
        "GPT2": "openai-community/gpt2",
        "BERT": "google-bert/bert-base-uncased",
        "LLAMA": "huggyllama/llama-7b",
    }
    fallback_dim_map = {
        "GPT2": 768,
        "BERT": 768,
        "LLAMA": 4096,
    }

    model_id = (
        getattr(config, "llm_model_id", None)
        or getattr(config, "model_id", None)
        or model_id_map.get(model_name)
    )

    if model_id:
        try:
            from transformers import AutoConfig
            try:
                hf_cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
            except Exception:
                hf_cfg = AutoConfig.from_pretrained(model_id, local_files_only=False)
            for attr in ("hidden_size", "n_embd", "d_model", "dim"):
                value = getattr(hf_cfg, attr, None)
                if value is not None:
                    return int(value)
        except Exception:
            pass

    if model_name in fallback_dim_map:
        return int(fallback_dim_map[model_name])
    return int(llm_dim)


class TimeLLMAdapter(BaseTSFMAdapter):
    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, TIME_LLM_HPARAMS, **kwargs)

        self.config.pred_len = int(
            getattr(self.config, "pred_len", getattr(self.config, "horizon", 0))
        )
        self.config.horizon = int(getattr(self.config, "horizon", self.config.pred_len))
        self.config.pred_len = self.config.horizon

        seq_len = int(getattr(self.config, "seq_len", 96))
        patch_len = int(getattr(self.config, "patch_len", 16))
        patch_len = max(1, min(patch_len, seq_len))
        stride = int(getattr(self.config, "stride", max(1, patch_len // 2)))
        stride = max(1, min(stride, patch_len))

        self.config.seq_len = seq_len
        self.config.patch_len = patch_len
        self.config.stride = stride

        if int(getattr(self.config, "label_len", 0)) <= 0:
            self.config.label_len = self.config.horizon

        self.config.norm = bool(getattr(self.config, "norm", False))

        self.config.llm_layers = int(getattr(self.config, "llm_layers", 6))
        llm_dim = int(getattr(self.config, "llm_dim", 4096))
        if "llm_dim" in kwargs and kwargs.get("llm_dim") is not None:
            self.config.llm_dim = llm_dim
        else:
            self.config.llm_dim = _infer_llm_dim(self.config, llm_dim)
        self.config.prompt_domain = int(getattr(self.config, "prompt_domain", 0))
        if not hasattr(self.config, "content"):
            self.config.content = ""

        self.config.enc_in = int(getattr(self.config, "enc_in", 1))
        self.config.dec_in = int(getattr(self.config, "dec_in", self.config.enc_in))
        self.config.c_out = int(getattr(self.config, "c_out", self.config.enc_in))

    def _on_shot_mode_resolved(self, mode: str) -> None:
        if mode in {"few_shot", "full_shot"}:
            self.config.lradj = "type1"


def time_llm_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=TimeLLMAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )


def tsfm_adapter(model_info: Type[object]) -> object:
    return time_llm_adapter(model_info)