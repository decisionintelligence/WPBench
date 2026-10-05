from ts_benchmark.baselines.xlinear.model.xlinear_model import xlinear_model
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
import torch

MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "embed_type": 0,
    "embed": "timeF",
    "features": "MS",
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "freq": "h",
    "factor": 1,
    "n_heads": 8,
    "activation": "gelu",
    "output_attention": False,
    "patch_len": 16,
    "stride": 8,
    "padding_patch": "end",
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "decomposition": 0,
    "kernel_size": 25,
    "individual": 0,
    "dropout": 0.05,
    "fc_dropout": 0.05,
    "head_dropout": 0.0,
    "moving_avg": 25,
    "pct_start": 0.3,
    "batch_size": 128,
    "lradj": "type3",
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 10,
    "loss": "MSE",
    "patience": 100,
    "t_ff": 1,
    "c_ff": 1,
    "t_dropout": 0.0,
    "c_dropout": 0.0,
    "embed_dropout": 0.1,
    "usenorm": 1,
    "norm": True,
    "use_onecycle_lr_init": True,
}


class Xlinear(DeepForecastingModelBase):
    """
    Xlinear adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the AmplifierModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(Xlinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "Xlinear"

    def _init_model(self):
        return xlinear_model(self.config)

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        # 交换一下目标列和需要预测列的位置（由于总是预测最后一列，这里也无需做过多的额外处理，把第一列放到最后一列就行）
        # 取出第一列放到最后一列即可
        # 输入应该是B,T,C？那这里需要把C维的第一列放到最后一列吗？如果是的话，下面的代码就可以了

        #input = torch.cat([input[:, 1:, :], input[:, :1, :]], dim=1)
        input = torch.cat([input[:, :, 1:], input[:, :, :1]], dim=2)
        output = self.model(input)

        return {"output": output}
