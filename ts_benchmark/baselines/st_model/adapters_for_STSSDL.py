from typing import Type

import torch.nn.functional as F

from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.st_model.utils.STSSDL_utils import ContrastiveLoss

# 假设这里导入了父类和你之前写的 utils
# from utils.stssdl_utils import generate_stssdl_features_df

# ST-SSDL 专属的默认超参数
STSSDL_HYPER_PARAMS = {
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.01,
    "num_epochs": 100,
    "loss": "MAE",  # 交通/风电通常用 MAE
    "adj_type": "symadj",
    "use_curriculum_learning": True,  # 开启课程学习
    "cl_decay_steps": 2000,  # 课程学习概率衰减步数
    # ... 其他 ST-SSDL 网络需要的参数 (如 e_layers, node_emb_dim 等)
    "rnn_units": 128,
    "embed_dim": 10,
    "num_layers": 1,
    "cheb_k": 3,
    "ycov_dim": 1,
    "prototype_num": 20,
    "prototype_dim": 64,
    "tod_embed_dim": 10,
    "use_STE": True,
    "adaptive_embedding_dim": 48,
    "node_embedding_dim": 20,
    "input_embedding_dim": 128,
    "contra_loss": "triplet",
    "temp": 1.0,
    "lamb_c": 0.01,
    "lamb_d": 0.01,

}


class STSSDLAdapter(DeepForecastingModelBase):
    """
    专门为 ST-SSDL 模型定制的 Adapter。
    继承自 DeepForecastingModelBase，主要劫持 forecast_fit 注入历史锚点特征，
    并重写 _process 适配自监督偏差损失和课程学习。
    """

    def __init__(self, model_name, model_class, **kwargs):
        # 传入 ST-SSDL 的专属超参
        super(STSSDLAdapter, self).__init__(STSSDL_HYPER_PARAMS, **kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.batches_seen = 0  # 用于记录当前训练了多少个 batch (给课程学习用)

    @property
    def model_name(self):
        return self._model_name

    def _init_model(self):
        return self.model_class(self.config)

    # ==========================================
    # 💡 核心 1：重写 forecast_fit (拦截并改造数据)
    # ==========================================
    # def forecast_fit(
    #         self,
    #         train_valid_data: pd.DataFrame,
    #         *,
    #         covariates: Optional[dict] = None,
    #         train_ratio_in_tv: float = 1.0,
    #         adj_mx: Optional[dict] = None,
    #         **kwargs,
    # ) -> "ModelBase":
    #     """
    #     Train the model.
    #     :param train_valid_data: Time series data used for training and validation.
    #     :param covariates: Additional external variables.
    #     :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
    #     :return: The fitted model object.
    #     """
    #     # 首先检查train_valid_data中有几组这样重复的数据
    #     if covariates is None:
    #         covariates = {}
    #     series_num = self.detect_series_groups(train_valid_data)
    #
    #     # 额外的情况：查看最后一个type的数字：
    #     last_col_name = train_valid_data.columns[-1]
    #     # print(f"Series shape: {data.shape}, Last column name: {last_col_name}")
    #     # 检查last_col_name是否符合type+数字+":"的格式
    #     if last_col_name is not None and isinstance(last_col_name, str) and "type" in last_col_name.lower():
    #         inferred = self._infer_series_number_from_col(last_col_name)
    #         if inferred is not None and inferred >= 1:
    #             series_num = inferred
    #
    #     self.config.num_nodes = series_num
    #     series_dim = train_valid_data.shape[-1] // series_num
    #     if series_num * series_dim != train_valid_data.shape[-1]:
    #         raise ValueError("Data columns cannot be evenly divided by series_num.")
    #
    #     # 单独筛选出其中的一列作为sample_pandas，用于确定时间周期等pandas相关的数值
    #     sample_train_valid_data = train_valid_data.iloc[:, :series_dim]
    #     # 注意，此内容需要重新进行重新的划分，考虑到我已经拥有了series_num，那么我的train_valid_data和exog_data都应该从(T*(C*N))->T*C*N
    #     train_valid_data = self.reshape_spatiotemporal(train_valid_data, series_num)[0]
    #     exog_data = covariates.get("exog", None)
    #     if exog_data is not None:
    #         exog_dim = exog_data.shape[-1] // series_num
    #         sample_exog_data = exog_data.iloc[:, :exog_dim]
    #         exog_data = self.reshape_spatiotemporal(exog_data, series_num)[0]
    #         train_valid_data = np.concatenate([train_valid_data, exog_data], axis=1)
    #         exog_dim = exog_data.shape[-2]
    #     else:
    #         exog_dim = 0
    #
    #     sample_train_valid_data = pd.concat([sample_train_valid_data, sample_exog_data],
    #                                         axis=1) if exog_data is not None else sample_train_valid_data
    #     if sample_train_valid_data.shape[1] == 1:
    #         train_drop_last = False
    #         self.single_forecasting_hyper_param_tune(sample_train_valid_data)
    #     else:
    #         train_drop_last = True
    #         self.multi_forecasting_hyper_param_tune(sample_train_valid_data)
    #
    #     self.config.series_dim = series_dim
    #     self.config.input_dim = series_dim + exog_dim
    #     self.config.output_dim = series_dim
    #     self.config.adj_mx = adj_mx
    #     self.config.series_num = series_num
    #     # 计算出有关STSSDL的图的系列指标：
    #     if self.model_name in ["STSSDL"]:
    #         adj_mx = load_adj(self.config.adj_mx, self.config.adj_type)
    #         self.config.adj_mx = adj_mx
    #     criterion = self._init_criterion()
    #     self.model = self._init_model()
    #     if self.config.fusion_method == "mlp":
    #         self.CovariateFusion = MLP(self.config)
    #     elif self.config.fusion_method == "cross_attention":
    #         self.CovariateFusion = CrossAttention(self.config)
    #     elif self.config.fusion_method == "conv":
    #         self.CovariateFusion = Conv(self.config)
    #     else:
    #         self.CovariateFusion = None
    #     device_ids = np.arange(torch.cuda.device_count()).tolist()
    #     if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
    #         self.model = nn.DataParallel(self.model, device_ids=device_ids)
    #         if self.CovariateFusion is not None:
    #             self.CovariateFusion = nn.DataParallel(
    #                 self.CovariateFusion, device_ids=device_ids
    #             )
    #     print(
    #         "----------------------------------------------------------",
    #         self.model_name,
    #     )
    #     config = self.config
    #     train_data, valid_data = train_val_split(
    #         train_valid_data, train_ratio_in_tv, config.seq_len
    #     )
    #     # 用于获取timestamp数据
    #     sample_train_data, sample_valid_data = train_val_split(
    #         sample_train_valid_data, train_ratio_in_tv, config.seq_len
    #     )
    #     train_data_l = train_data.shape[0]
    #     valid_data_l = valid_data.shape[0]
    #
    #     # 分别 fit 两个 scaler
    #     if exog_dim > 0:
    #         # Fit scaler1 for series data
    #         self.scaler1.fit(rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c"))
    #         # Fit scaler2 for exog data
    #         self.scaler2.fit(rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c"))
    #
    #         # self.scaler.fit(train_data.values)
    #         if config.norm:
    #             scaled_series = self.scaler1.transform(
    #                 rearrange(train_data[:, :series_dim, :], "l c n -> (l n) c")
    #             )
    #             train_series = rearrange(scaled_series, "(l n) c -> l c n", l=train_data_l)
    #
    #             scaled_exog = self.scaler2.transform(
    #                 rearrange(train_data[:, series_dim:, :], "l c n -> (l n) c")
    #             )
    #             train_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=train_data_l)
    #
    #             train_data = np.concatenate((train_series, train_exog), axis=1)
    #             """
    #             train_data = pd.DataFrame(
    #                 # self.scaler.transform(train_data.values),
    #                 final_train_data,
    #                 columns=train_data.columns,
    #                 index=train_data.index,
    #             )
    #             """
    #     else:
    #         # Only series data, use scaler1
    #         self.scaler1.fit(rearrange(train_data, "l c n -> (l n) c"))
    #         if config.norm:
    #             scaled_data = self.scaler1.transform(
    #                 rearrange(train_data, "l c n -> (l n) c")
    #             )
    #             train_data = rearrange(scaled_data, "(l n) c -> l c n", l=train_data_l)
    #
    #             """
    #             train_data = pd.DataFrame(
    #                 self.scaler1.transform(train_data.values),
    #                 columns=train_data.columns,
    #                 index=train_data.index,
    #             )
    #             """
    #
    #     if train_ratio_in_tv != 1:
    #         if config.norm:
    #             if exog_dim > 0:
    #                 # Scale validation series data
    #                 scaled_series = self.scaler1.transform(
    #                     rearrange(valid_data[:, :series_dim, :], "l c n -> (l n) c")
    #                 )
    #                 valid_series = rearrange(scaled_series, "(l n) c -> l c n", l=valid_data_l)
    #
    #                 # Scale validation exog data
    #                 scaled_exog = self.scaler2.transform(
    #                     rearrange(valid_data[:, series_dim:, :], "l c n -> (l n) c")
    #                 )
    #                 valid_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=valid_data_l)
    #
    #                 # Concatenate scaled data
    #                 valid_data = np.concatenate(
    #                     (valid_series, valid_exog), axis=1
    #                 )
    #
    #                 """
    #                 valid_data = pd.DataFrame(
    #                     final_valid_data,
    #                     columns=valid_data.columns,
    #                     index=valid_data.index,
    #                 )
    #                 """
    #             else:
    #                 scaled_data = self.scaler1.transform(
    #                     rearrange(valid_data, "l c n -> (l n) c")
    #                 )
    #                 valid_data = rearrange(scaled_data, "(l n) c -> l c n", l=valid_data_l)
    #
    #                 """
    #                 valid_data = pd.DataFrame(
    #                     self.scaler1.transform(valid_data.values),
    #                     columns=valid_data.columns,
    #                     index=valid_data.index,
    #                 )
    #                 """
    #         # 额外添加一个模型名号，然后生成即可
    #         valid_dataset, valid_data_loader = forecasting_data_provider(
    #             valid_data,
    #             config,
    #             timeenc=1,
    #             batch_size=config.batch_size,
    #             shuffle=True,
    #             drop_last=False,
    #             sample_timestamp=sample_valid_data,
    #             model_name=self.model_name,
    #         )
    #
    #     train_dataset, self.train_data_loader = forecasting_data_provider(
    #         train_data,
    #         config,
    #         timeenc=1,
    #         batch_size=config.batch_size,
    #         shuffle=True,
    #         drop_last=train_drop_last,
    #         sample_timestamp=sample_train_data,
    #         model_name=self.model_name,
    #     )
    #     # Define optimizer
    #     optimizer = self._init_optimizer(CovariateFusion=self.CovariateFusion)
    #
    #     if config.use_amp == 1:
    #         scaler = torch.cuda.amp.GradScaler()
    #
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    #     self.early_stopping = self._init_early_stopping()
    #     self.model.to(device)
    #     if self.CovariateFusion is not None:
    #         self.CovariateFusion.to(device)
    #     total_params = sum(
    #         p.numel() for p in self.model.parameters() if p.requires_grad
    #     )
    #     print(f"model total trainable parameters:{total_params}")
    #     if self.CovariateFusion is not None:
    #         total_params += sum(
    #             p.numel() for p in self.CovariateFusion.parameters() if p.requires_grad
    #         )
    #     print(f"Total trainable parameters: {total_params}")
    #
    #     for epoch in range(config.num_epochs):
    #         self.model.train()
    #         if self.CovariateFusion is not None:
    #             self.CovariateFusion.train()
    #         # for input, target, input_mark, target_mark in train_data_loader:
    #         for i, (input, target, input_mark, target_mark) in enumerate(
    #                 self.train_data_loader
    #         ):
    #             optimizer.zero_grad()
    #             input, target, input_mark, target_mark = (
    #                 input.to(device),
    #                 target.to(device),
    #                 input_mark.to(device),
    #                 target_mark.to(device),
    #             )
    #             # decoder input
    #             exog_future = target[:, -config.horizon:, series_dim:].to(device)
    #             out_loss = self._process(
    #                 input, target, input_mark, target_mark, exog_future
    #             )
    #             additional_loss = 0
    #             output = out_loss["output"]
    #             if "additional_loss" in out_loss:
    #                 additional_loss = out_loss["additional_loss"]
    #             if len(output.shape) == 3:
    #                 target = target[:, -config.horizon:, :series_dim]
    #                 output = output[:, -config.horizon:, :series_dim]
    #             if len(output.shape) == 4:
    #                 # 此时output维度为B,N,T,C
    #                 target = target[:, :, -config.horizon:, :series_dim]
    #                 output = output[:, :, -config.horizon:, :series_dim]
    #                 B, N, H, C = target.shape
    #
    #                 # 把前两个维度合并
    #                 target = target.reshape(B * N, H, C)
    #                 output = output.reshape(B * N, H, C)
    #             if (
    #                     self.config.fusion_method == "mlp"
    #                     or self.config.fusion_method == "conv"
    #                     or self.config.fusion_method == "cross_attention"
    #             ) and self.CovariateFusion is not None:
    #                 output = self.CovariateFusion(exog_future, output)
    #             output, target = self._post_process(output, target)
    #
    #             loss = criterion(output, target)
    #             # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #             total_loss = loss + additional_loss
    #
    #             if config.use_amp == 1:
    #                 scaler.scale(total_loss).backward()
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #             else:
    #                 total_loss.backward()
    #                 optimizer.step()
    #
    #             if self.config.lradj == "TST":
    #                 self._adjust_lr(optimizer, epoch + 1, config)
    #
    #         if train_ratio_in_tv != 1:
    #             valid_loss = self.validate(valid_data_loader, series_dim, criterion)
    #             improved = self.early_stopping(valid_loss, self.model)
    #             if improved:
    #                 if self.CovariateFusion is not None:
    #                     self.check_point = self.save_checkpoint(
    #                         {
    #                             "Model": self.model,
    #                             "CovariateFusion": self.CovariateFusion,
    #                         }
    #                     )
    #                 else:
    #                     self.check_point = self.save_checkpoint({"Model": self.model})
    #             if self.early_stopping.early_stop:
    #                 break
    #
    #         if self.config.lradj != "TST":
    #             self._adjust_lr(optimizer, epoch + 1, config)

    # ==========================================
    # 💡 核心 2：重写 _process (拆解 C=3 的通道，计算辅助 Loss)
    # ==========================================
    def _process(self, input, target, input_mark, target_mark, exog_future=None):
        """
        ST-SSDL 的 forward 需要: x, x_cov, x_his, y_cov, labels, batches_seen
        我们的 input 现在通道 C=3，分别是：[value, timeofday, flow_y]
        """
        # 1. 维度对齐与拆解 (假设 input 形状为 B, N, T, C)
        # 如果底层传过来的是 B, T, N, C，请自行 transpose(1, 2)

        # 提取当前特征 X^c
        x = input[..., 0:1].float()
        # 提取时间标签特征 x_cov (比如 timeofday)
        x_cov = input[..., 1:2].float()
        # 提取历史锚点 X^a
        x_his = input[..., 2:3].float()

        # 目标特征 (Y) 的时间标签 y_cov
        y_cov = target[..., 1:2].float()
        # 真实的标签 labels (纯数值，给 Teacher Forcing 用)
        labels = target[..., 0:1].float()

        # 2. 更新 batches_seen (仅在训练阶段递增)
        if self.model.training:
            self.batches_seen += 1

        # 维度转换：从B,N,T,C到B,T,N,C
        x = x.permute(0, 2, 1, 3).contiguous()
        x_cov = x_cov.permute(0, 2, 1, 3).contiguous()
        x_his = x_his.permute(0, 2, 1, 3).contiguous()
        y_cov = y_cov.permute(0, 2, 1, 3).contiguous()
        labels = labels.permute(0, 2, 1, 3).contiguous()

        # 3. 执行前向传播
        output, query, pos, neg, mask, query_simi, pos_simi = self.model(
            x=x,
            x_cov=x_cov,
            x_his=x_his,
            y_cov=y_cov,
            labels=labels,
            batches_seen=self.batches_seen
        )
        contrastive_loss = ContrastiveLoss(contra_loss=self.config.contra_loss, mask=mask, temp=self.config.temp)

        loss_c = contrastive_loss.calculate(query[0], pos[0], neg[0], mask[0])
        loss_d = F.l1_loss(query_simi.detach(), pos_simi)
        additional_loss = self.config.lamb_c * loss_c + self.config.lamb_d * loss_d

        # output维度对齐
        output = output.permute(0, 2, 1, 3)

        # 返回约定的字典格式给父类的训练循环
        out_loss = {
            "output": output,
            "additional_loss": additional_loss
        }
        return out_loss


# 工厂函数生成 Adapter
def stssdl_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    def model_factory(**kwargs) -> STSSDLAdapter:
        return STSSDLAdapter(model_info.__name__, model_info, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    }
