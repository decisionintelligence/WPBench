from typing import Type

import numpy as np
import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_model.utils.PatchSTG_utils import _load_spatial_managed_indices

PatchSTG_HYPER_PARAMS = {
    'tod': 96, # time of day(N_d)
    'dow': 7,   # N_w
    'input_dims': 64, # d_e
    'node_dims': 64,  # spatio embedding(d_s)
    'tod_dims': 32,  # d_d
    'dow_dims': 32,  # d_w

    'tem_patchsize': 12,
    'tem_patchnum': 1,
    'spa_patchnum': 512,
    'spa_patchsize': 2,
    'factors': 32, # KDTree leaf node merge factor
    'layers': 5,    # Dual attention
    'batch_size': 16,
    'lr': 0.002,
    'max_epoch': 50,
}

class PatchSTGAdapter(DeepForecastingModelBase):
    def __init__(self, model_name, model_class, **kwargs):
        super(PatchSTGAdapter, self).__init__(PatchSTG_HYPER_PARAMS, **kwargs)
        self._model_name = model_name
        self.model_class = model_class

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        seq_len = int(self.config.seq_len)
        tem_patchsize = int(self.config.tem_patchsize)
        if seq_len % tem_patchsize != 0:
            raise ValueError(
                f"PatchSTG expects seq_len to be divisible by tem_patchsize, got seq_len={seq_len}, tem_patchsize={tem_patchsize}"
            )
        tem_patchnum = seq_len // tem_patchsize
        if int(self.config.tem_patchnum) != tem_patchnum:
            self.config.tem_patchnum = tem_patchnum
        # from config.geo_data (DeepForecastingModelBase.forecast_fit)
        locations = getattr(self.config, "geo_data", None)
        # similarity between nodes
        adj = getattr(self.config, "adj_mx", None)
        if locations is not None and adj is not None:
            spa_patchnum = self.config.spa_patchnum
            spa_patchsize = self.config.spa_patchsize
            recurtimes = int(np.log2(spa_patchnum))
            ori_parts_idx, reo_parts_idx, reo_all_idx = _load_spatial_managed_indices(
                locations, adj, recurtimes, spa_patchsize
            )
            self.config.ori_parts_idx = ori_parts_idx
            self.config.reo_parts_idx = reo_parts_idx
            self.config.reo_all_idx = reo_all_idx
        return self.model_class(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # input/target_mark:last dim: C_mark
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information, 2: time of day, day of week
        # T_in: seq_len ,T_out: horizon
        if input.dim() != 4:
            raise ValueError(f"PatchSTG expects input with 4 dims (B, N, T, C), got {input.shape}")
        if input.shape[-1] < 1:
            raise ValueError(f"PatchSTG expects at least 1 value channel, got last dim {input.shape[-1]}")
        x = input[..., 0:1] # [B,N,T_in,1]
        if input_mark is None:
            raise ValueError("PatchSTG requires historical temporal marks from input_mark.")
        if input_mark.dim() == 4:
            te = input_mark[:, 0, :, :] # [B,T_in,C_mark]
        else:
            te = input_mark
        if te.shape[1] != input.shape[2]:
            raise ValueError(
                f"PatchSTG expects input_mark T to match input T, got {te.shape[1]} and {input.shape[2]}"
            )
        x = x.permute(0, 2, 1, 3).contiguous()
        B, _, N, _ = x.shape
        te = te.unsqueeze(2).expand(-1, -1, N, -1)
        output = self.model(x, te)
        # pred_y: [B,T_out,N,1]
        output = output.permute(0, 2, 1, 3).contiguous() #-> [B,N,T_out,1]
        return {"output": output}

def patchstg_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    def model_factory(**kwargs) -> PatchSTGAdapter:
        return PatchSTGAdapter(model_info.__name__, model_info, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    }


