from typing import Type

from ts_benchmark.baselines.stfm.adapters_for_stfm import (
    BaseSTFMAdapter,
    GENERIC_STFM_HPARAMS,
    generate_model_factory,
)
from ts_benchmark.baselines.stfm.models.factost_utp_io import (
    build_factost_utp_inputs,
    restore_factost_utp_output,
)

FACTOST_UTP_HPARAMS = {
    **GENERIC_STFM_HPARAMS,
    "is_spatial": True,
    "label_len": 0,
    "norm": True,
    # Upstream exp_factost_utp.sh default profile: tiny_4utp.
    "patch_len": 16,
    "stride": 16,
    "factost_n_layers": 3,
    "factost_n_heads": 4,
    "factost_d_model": 256,
    "factost_d_ff": 1024,
    "factost_dropout": 0.2,
}


class FactoSTUTPAdapter(BaseSTFMAdapter):
    """Benchmark adapter for channel-independent FactoST UTP zero-shot."""

    def __init__(self, model_name, model_class, **kwargs):
        super().__init__(model_name, model_class, FACTOST_UTP_HPARAMS, **kwargs)
        self.config.is_spatial = True
        self.config.label_len = 0

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        del target, input_mark, target_mark, exog_future
        x_utp, layout = build_factost_utp_inputs(input)
        output = self.model(x_utp)
        return {"output": restore_factost_utp_output(output, layout)}
    

def factost_utp_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")
    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        adapter_cls=FactoSTUTPAdapter,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
        },
    )
