from ts_benchmark.baselines.st_model.models.STDN import STDNModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_model.utils.matrix_calculate import cal_lape
import torch
import torch.nn as nn

# model hyper params
MODEL_HYPER_PARAMS = {
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 100,
    "num_workers": 0,
    "loss": "MAE",
    "use_future_exog": False,
    "L": 2,
    "K": 16,
    "d": 8,
    "order":3,
    "reference":3,
    "seq_len": 12,
    "pred_len":12,
    "out_channels": 1
}


class STDN(DeepForecastingModelBase):
    """
    DUET adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of the DUETModel.
        _adjust_lr：Adjusts the learning rate of the optimizer based on the current epoch and configuration.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(STDN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "STDN"

    def _init_model(self):
        model = STDNModel( self.configs, bn_decay=0.1)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        adj_mx = self.config.adj_mx
        lpls = cal_lape(adj_mx.copy())
        TE = torch.cat([input_mark, target_mark[:, -self.config.horizon:, :]], dim=1)

        output, loss_importance = self.model(input,TE,lpls)
        out_loss = {"output": output}
        if self.model.training:
            out_loss["additional_loss"] = loss_importance
        return out_loss
