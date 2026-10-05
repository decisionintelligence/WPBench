from typing import Type, Dict

import torch

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_model.utils.STWave_utils import disentangle

# model hyper params
MODEL_HYPER_PARAMS = {
    # "top_k": 5,
    # "enc_in": 1,
    # "dec_in": 1,
    # "c_out": 1,
    # "e_layers": 2,
    # "d_layers": 1,
    # "d_model": 512,
    # "d_ff": 2048,
    # "embed": "timeF",
    # "freq": "h",
    # "lradj": "type1",
    # "moving_avg": 25,
    # "num_kernels": 6,
    # "factor": 1,
    # "n_heads": 8,
    # "seg_len": 6,
    # "win_size": 2,
    # "activation": "softmax",
    # "output_attention": 0,
    # "patch_len": 16,
    # "stride": 8,
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MSE",
    # "itr": 1,
    # "distil": True,
    # "patience": 3,
    # "p_hidden_dims": [128, 128],
    # "p_hidden_layers": 2,
    # "mem_dim": 32,
    # "conv_kernel": [12, 16],
    # "anomaly_ratio": 1.0,
    # "down_sampling_windows": 2,
    # "channel_independence": True,
    # "down_sampling_layers": 3,
    # "down_sampling_method": "avg",
    # "decomp_method": "moving_avg",
    # "use_norm": True,
    # "parallel_strategy": "DP",
    # "task_name": "short_term_forecast",
    "use_future_exog": False,
    "covariate_dim": 6,
    # TGGC
    "rnn_units": 64,
    "embed_dim": 10,
    "num_layers": 2,
    "cheb_order": 3,
    "leaky_rate": 0.2,
    "coe_a": 1,
    "coe_b": 1,
    "order": 4,
    "gconv": "gegen",
    "activation": "softmax",
    "non_linear": "linear",
    "Fouropt": "FB",
    "attention_set": "time",
    "modes": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "stack_cnt": 2,

    # STWave
    "heads": 8,
    "dims": 16,
    "samples": 1,
    "wave": "coif1",
    "level": 1,

    # STSSDL
    "adj_type": "symadj",

    # STID
    "node_dim": 32,
    "input_dim": 3,
    "embed_dim": 32,
    "temp_dim_tid": 32,
    "temp_dim_diw": 32,
    "day_of_week_size": 7,
    "if_T_i_D": True,
    "if_D_i_W": True,
    "if_node": True,

    # BigST
    "bigst_hidden_dim": 32,
    "bigst_node_dim": 32,
    "bigst_time_dim": 32,
    "bigst_num_layers": 3,
    "bigst_tau": 1.0,
    "bigst_random_feature_dim": 64,
    "bigst_use_residual": True,
    "bigst_use_bn": True,
    "bigst_use_spatial": False,


    # "skip_dim": 256,
    # "lape_dim": 8,
    # "geo_num_heads": 4,
    # "sem_num_heads": 4,
    # "t_num_heads": 2,
    #  "mlp_ratio": 4,
    #  "qkv_bias": True,
    #  "attn_dropout": 0,
    #  "s_attn_size": 3,
    #  "t_attn_size": 3,
    #  "enc_depth": 6,
    #  "type_ln": "pre",
    # "type_short_path": "hop",
    # "add_time_in_day": True,
    # "add_day_in_week": True,
    #
    # "far_mask_delta": 5,
    # "dtw_delta": 5



}


class STmodelAdapter(DeepForecastingModelBase):
    """
    Time Series Library adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _init_criterion_and_optimizer: Defines the loss function and optimizer.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, model_name, model_class, **kwargs):
        super(STmodelAdapter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self._model_name = model_name
        self.model_class = model_class

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        return self.model_class(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # decoder input
        dec_input = torch.zeros_like(target[:, -self.config.horizon :, :]).float()
        dec_input = (
            torch.cat([target[:, : self.config.label_len, :], dec_input], dim=1)
            .float()
            .to(input.device)
        )
        if (
            self._model_name == "TiDE"
            or self._model_name == "TemporalFusionTransformer"
        ):
            output = self.model(input, input_mark, exog_future, target_mark)
            out_loss = {"output": output}
        elif self.model_name == "TGGC":
            output, attention, backcast = self.model(input)
            out_loss = {"output": output}
            out_loss["attention"] = attention
            out_loss["backcast"] = backcast
        # elif self.model_name == "PDFormer":
        #     adj_mx = self.config.adj_mx
        #     lpls_np = cal_lape(adj_mx.copy())
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     lpls = torch.tensor(lpls_np, dtype=torch.float32).to(device)
        #     output = self.model(input,lpls)
        #     out_loss = {"output": output}
        elif self.model_name == "STWave":
            # if input.shape[-1] > 1:
            #     input = input[..., 0:1]
            # if target.shape[-1] > 1:
            #     target = target[..., 0:1]
            # 1. 先把 (B, N, T, C) 转成原作者需要的 (B, T, N, C)
            input_st = input.transpose(1, 2).contiguous()
            target_st = target.transpose(1, 2).contiguous()

            # 2. 极其关键：转成 NumPy 数组！
            inp_np = input_st.cpu().numpy()
            tgt_np = target_st.cpu().numpy()

            # 3. 把 Numpy 数组传进去，就不会报 transpose 错误了
            XL, XH = disentangle(inp_np, self.config.wave, self.config.level)
            YL, _ = disentangle(tgt_np, self.config.wave, self.config.level)

            # 4. 算完之后，重新包回 Tensor 放回 GPU
            XL = torch.from_numpy(XL).float().to(input.device)
            XH = torch.from_numpy(XH).float().to(input.device)
            YL = torch.from_numpy(YL).float().to(input.device)

            if target_mark.dim() == 4:  # (B, N, T, C)
                TE_pred = target_mark[:, :, -self.config.horizon:, :]
            else:
                # 记得修改process部分的调用
                TE_pred = target_mark[:, -self.config.horizon:, :]

                # 检查维度：如果 input_mark 是 3 维而 TE_pred 是 4 维（或反之），需要对齐
            if input_mark.dim() == 4:  # (B, N, T, C)
                input_mark = input_mark[:, 0, :, :]  # 取其中一个节点的即可，因为时间特征共享
            if TE_pred.dim() == 4:
                TE_pred = TE_pred[:, 0, :, :]

            TE = torch.cat([input_mark, TE_pred], dim=1)
            y_hat, y_hat_l = self.model(XL, XH, TE)

            y_hat = y_hat.transpose(1, 2)
            y_hat_l = y_hat_l.transpose(1, 2)
            YL = YL.transpose(1, 2)
            out_loss = {"output": y_hat, "y_hat_l": y_hat_l, "YL": YL}
        elif self.model_name == "STID":
            if self.config.series_dim != 1:
                raise ValueError(
                    f"STID currently expects series_dim == 1, but got {self.config.series_dim}."
                )

            history_value = input[..., : self.config.series_dim]
            history_value = history_value.transpose(1, 2).contiguous()

            if input_mark.dim() == 4:
                stid_mark = input_mark[:, 0, :, :]
            else:
                stid_mark = input_mark

            if stid_mark.shape[-1] < 2:
                raise ValueError(
                    f"STID expects at least two timestamp features [timeofday, dayofweek], but got shape {stid_mark.shape}."
                )

            stid_mark = stid_mark[..., -2:]
            stid_mark = stid_mark.unsqueeze(2).expand(
                -1, -1, history_value.shape[2], -1
            )
            history_data = torch.cat([history_value, stid_mark], dim=-1)

            output = self.model(history_data)
            output = output.transpose(1, 2).contiguous()
            out_loss = {"output": output}
        else:
            output = self.model(input, input_mark, dec_input, target_mark)
            out_loss = {"output": output}
        return out_loss


def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    Generate model factory information for creating Transformer Adapters model adapters.

    :param model_name: Model name.
    :param model_class: Model class.
    :param required_args: The required parameters for model initialization.
    :return: A dictionary containing model factories and required parameters.
    """

    def model_factory(**kwargs) -> STmodelAdapter:
        """
        Model factory, used to create TransformerAdapter model adapter objects.

        :param kwargs: Model initialization parameters.
        :return:  Model adapter object.
        """
        return STmodelAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def stmodel_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )
