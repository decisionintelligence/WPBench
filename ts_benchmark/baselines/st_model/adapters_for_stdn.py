from ts_benchmark.baselines.st_model.models.STDN import STDNModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_model.utils.matrix_calculate import cal_lape
import torch
import torch.nn as nn
from typing import Type, Dict

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
    #  修改构造函数：显式接收 model_name
    def __init__(self, model_name: str, **kwargs):
        # 将配置字典 MODEL_HYPER_PARAMS 传给父类
        super(STDN, self).__init__(MODEL_HYPER_PARAMS, **kwargs)
        self._name = model_name

    @property
    def model_name(self):
        return self._name

    def _init_model(self):
        # ⚠️ 注意：基类属性是 self.config (单数)
        model = STDNModel(self.config, bn_decay=0.1)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def _process(self, input, target, input_mark, target_mark, exog_future=None):

        # 1. 获取邻接矩阵
        adj_mx = self.config.adj_mx

        # 2. 计算拉普拉斯位置编码 (返回的是 numpy.ndarray)
        lpls_np = cal_lape(adj_mx.copy())

        # 3. 🚀 关键修正：转换为 Tensor 并移动到设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpls = torch.tensor(lpls_np, dtype=torch.float32).to(device)

        effective_his_len = int(getattr(self.config, "horizon", self.config.pred_len))
        if input.dim() == 4:
            input = input[:, :, -effective_his_len:, :]
        else:
            input = input[:, -effective_his_len:, :]

        # 1. 🚀 关键修复：确保 input_mark 和 TE_pred 维度对齐
        # 强制取最后 horizon 长度的预测时间戳
        if target_mark.dim() == 4:  # (B, N, T, C)
            TE_pred = target_mark[:, :, -self.config.horizon:, :]
        else:
            # 记得修改process部分的调用
            TE_pred = target_mark[:,-self.config.horizon:, :]

        # 检查维度：如果 input_mark 是 3 维而 TE_pred 是 4 维（或反之），需要对齐
        if input_mark.dim() == 4:  # (B, N, T, C)
            input_mark = input_mark[:, 0, -effective_his_len:, :]  # 取其中一个节点的即可，因为时间特征共享
        else:
            input_mark = input_mark[:, -effective_his_len:, :]
        if TE_pred.dim() == 4:
            TE_pred = TE_pred[:, 0, :, :]

        # 2. 现在它们的时间长度分别是 24 (his) 和 24 (pred)，进行拼接
        # 最终 TE 形状应为 (Batch, 48, 2)
        TE = torch.cat([input_mark, TE_pred], dim=1)

        # 5. 调用模型
        stdn_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(stdn_model, "num_his"):
            stdn_model.num_his = effective_his_len

        output = self.model(input, TE, lpls)
        output = output.permute(0, 2, 1, 3)  # (B, N, T, C)
        out_loss = {"output": output}

        return out_loss

#  修改 Factory 函数：只保留 model_name
def generate_model_factory(model_name: str, required_args: dict) -> Dict:
    def model_factory(**kwargs) -> STDN:
        # 这里只传一个位置参数 model_name，对应 STDN.__init__
        return STDN(model_name, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }

#  修改 Adapter 入口
def stdn_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )
