import copy
import logging
import os
from typing import Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from pandas import DatetimeIndex, TimedeltaIndex
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from ts_benchmark.baselines.external_scaler import (
    fit_spatial_external_scaler,
    inverse_transform_flat_bntc,
    inverse_transform_wide_2d,
    resolve_external_scaler_mode,
    transform_flat_bntc,
    transform_stssdl_exog_flat_bntc,
    transform_stssdl_exog_tcn,
    transform_tcn,
)
from ts_benchmark.baselines.graph_utils import loadGraph
from ts_benchmark.baselines.utils import EarlyStopping, adjust_learning_rate
from ts_benchmark.baselines.utils import (
    forecasting_data_provider,
    train_val_split,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import (
    detect_repeating_channel_groups,
    infer_series_number,
    infer_series_number_from_col_name,
    split_time,
)
from ts_benchmark.baselines.utils import MLP, Conv, CrossAttention
# from ts_benchmark.baselines.st_model.utils.STSSDL_utils import load_adj

logger = logging.getLogger(__name__)

# Default hyper parameters
DEFAULT_HYPER_PARAMS = {
    "use_amp": 0,
    "loss": "MSE",
    "batch_size": 256,
    "lradj": "type3",
    "lr": 0.0001,
    "num_workers": 0,
    "patience": 10,
    "num_epochs": 100,
    "adj_lr_in_epoch": True,
    "adj_lr_in_batch": False,
    "parallel_strategy": None,
    "fusion_method": "",
    "covariate_dim": 6,
    "cross_attention_head": 12,
    "cross_attention_dropout": 0.1,
    "cross_attention_factor": 2,
    "conv_dropout": 0.1,
    "alpha_cov": 1.0,
    "mlp_hidden_dims": 64,
    "is_spatial": False,
    "external_scaler_mode": "pooled_target",
    "heads": 8,
    "dims": 16,
    "tem_graph": os.environ.get(
        "WPBENCH_DEFAULT_TEM_GRAPH",
        os.path.abspath(os.path.join(
            __file__, "..", "..", "..", "data", "graphs", "tem_graph",
            "stwave_tfb-gefcom2012_N7.npy",
        )),
    ),
}


class Config:
    def __init__(self, model_config, **kwargs):

        for key, value in DEFAULT_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in model_config.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

        if hasattr(self, "horizon"):
            logger.warning(
                "The model parameter horizon is deprecated. Please use pred_len."
            )
            setattr(self, "pred_len", self.horizon)


class DeepForecastingModelBase(ModelBase):
    """
    Base class for deep learning model in forecasting tasks, inherited from ModelBase.

    This class provides a framework and default functionalities for adapters in time series forecasting tasks,
    including model initialization, configuration of loss functions and optimizers, data processing,
    learning rate adjustment, save checkpoints and early stopping mechanisms.

    Subclasses must implement _init_model and _process methods to define specific data processing and modeling logic.

    """

    def __init__(self, model_config, **kwargs):
        super(DeepForecastingModelBase, self).__init__()
        self.config = Config(model_config, **kwargs)
        # self.scaler = StandardScaler()
        self.scaler1 = StandardScaler()
        self.scaler2 = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len
        self.check_point = None

    def _external_scaler_mode(self) -> str:
        return resolve_external_scaler_mode(
            getattr(self.config, "external_scaler_mode", "pooled_target")
        )

    def _init_model(self):
        """
        Initialize the model.

        This method is intended to be implemented by subclasses to initialize the specific model.
        The current implementation raises a NotImplementedError to indicate that this method should
        be overridden in subclasses.

        :return: The actual model object. The specific type of the return value should be defined by subclasses.
        """
        raise NotImplementedError("model must be implemented.")

    def _adjust_lr(self, optimizer, epoch, config):
        """
        Adjusts the learning rate of the optimizer based on the current epoch and configuration.

        This method is typically called to update the learning rate according to a predefined schedule.

        :param optimizer: The optimizer for which the learning rate will be adjusted.
        :param epoch: The current training epoch used to calculate the new learning rate.
        :param config: Configuration object containing parameters that control learning rate adjustment.
        """
        adjust_learning_rate(optimizer, epoch, config)

    def save_checkpoint(self, models):
        """
        Save the model checkpoint.

        This function saves the model's state dictionary (state_dict) to be used
        for restoring the model at a later time. A deep copy of the state_dict is returned.

        Parameters:
        - model (torch.nn.Module): The current instance of the model being trained.

        Returns:
        - OrderedDict: A deep copy of the model's state_dict, which can be used to restore
          the model's parameters in the future.
        """
        return {key: copy.deepcopy(model.state_dict()) for key, model in models.items()}

    def _init_criterion(self):
        """
        Initializes the task loss function.

        Supports MSELoss, L1Loss (MAE), and HuberLoss depending on `self.config.loss`.
        """
        if self.config.loss == "MSE":
            return nn.MSELoss()
        elif self.config.loss == "MAE":
            return nn.L1Loss()
        else:
            return nn.HuberLoss(delta=0.5)

    def _init_optimizer(self, CovariateFusion=None):
        """
        Initializes the optimizer using Adam.

        If `self.CovariateFusion` exists, creates parameter groups with separate learning rates.
        """
        if hasattr(self, "CovariateFusion") and self.CovariateFusion is not None:
            return optim.Adam(
                [
                    {"params": self.model.parameters(), "lr": self.config.lr},
                    {"params": self.CovariateFusion.parameters(), "lr": self.config.lr},
                ]
            )
        else:
            return optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _init_scheduler(self, optimizer, train_data_loader):
        if getattr(self.config, "use_onecycle_lr_init", False):
            return lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                steps_per_epoch=len(train_data_loader),
                pct_start=getattr(self.config, "pct_start", 0.3),
                epochs=self.config.num_epochs,
                max_lr=self.config.lr,
            )
        # del optimizer, train_data_loader
        return None

    def _process(self, input, target, input_mark, target_mark, exog_future=None, ):
        """
        A method that needs to be implemented by subclasses to process data and model, and calculate additional loss.

        This method's purpose is to serve as a template method, defining a standard process for data processing
        and modeling, as well as calculating any additional losses. Subclasses should implement specific processing
        and calculation logic based on their own needs.

        Parameters:
        - input: The input data, the specific form and meaning depend on the implementation of the subclass.
        - target: The target data, used in conjunction with input data for processing and loss calculation.
        - input_mark: Marks or metadata for the input data, assisting in data processing or model training.
        - target_mark: Marks or metadata for the target data, similarly assisting in data processing or model training.
        - exog_future: Exogenous future data, used in conjunction with input data for processing.

        Returns:
        - dict: A dictionary containing at least one key:
            - 'output' (necessary): The model output tensor.
            - 'additional_loss' (optional): An additional loss if it exists.

        Raises:
        - NotImplementedError: If the subclass does not implement this method, a NotImplementedError will be raised
                               when calling this method.
        """
        raise NotImplementedError("Process must be implemented")

    def _post_process(self, output, target):
        """
        Performs post-processing on the output and target data.

        This function is designed to process the output and target data after the model's forward computation,
        and return them directly in this example. The specific post-processing logic may include, but is not limited to,
        data format conversion, dimensionality matching, data type conversion, etc.

        Parameters:
        - output: The output data from the model, with no specific data format or type assumed.
        - target: The target data, which is the expected result, also without a fixed data format or type.

        Returns:
        - output: The output data after post-processing, which in this case is the same as the input.
        - target: The target data after post-processing, which in this case is the same as the input.
        """
        return output, target

    def _init_early_stopping(self):
        """
        Initializes the early stopping strategy for training.

        This function is used to create an instance of EarlyStopping, which helps prevent overfitting
        during model training by halting the training process when the validation performance
        does not improve for a specified number of consecutive iterations.

        Parameters:
        None directly, but it uses self.config.patience as the patience parameter for EarlyStopping.

        Returns:
        An instance of EarlyStopping, which monitors the model's performance metrics and determines
        when to stop the training.
        """
        return EarlyStopping(patience=self.config.patience)

    # 判断时空数据N维的数值
    def detect_series_groups(self, df: pd.DataFrame):
        """
        检测 DataFrame 列名中的重复通道分组模式。
        返回: (series_number, group_size, base_names, group_ids)
        - series_number: 组的个数
        - group_size: 每组的通道数
        - base_names: 基础通道名列表（第一组）
        - group_ids: 与列一一对应的组 id 列表
        """
        return detect_repeating_channel_groups(df)

    def reshape_spatiotemporal(self, train_valid_data: pd.DataFrame,
                               series_num: int):
        """
        将 (T, C*N) 的 DataFrame 转换为 (T, C, N) 的 numpy 数组，
        并可选地拼接 exog_data（外生变量）。

        参数:
        - train_valid_data: DataFrame, shape = (T, C*N)
        - series_num: int, 节点数 N
        - group_size: int, 每组通道数 C
        - exog_data: DataFrame 或 None, 外生变量，shape = (T, exog_dim*N)

        返回:
        - data_np: numpy.ndarray, shape = (T, C+exog_dim, N)
        - exog_dim: int, 当前变量的通道数
        """
        data_np = train_valid_data.values
        T = data_np.shape[0]
        group_size = data_np.shape[1] // series_num
        if data_np.shape[1] % series_num != 0:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        # 转换主数据
        data_np = rearrange(data_np, "t (n c) -> t c n", n=series_num)
        return data_np, group_size

    @property
    def model_name(self):
        return "DeepForecastingModelBase"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def multi_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):

        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()
            index : DatetimeIndex or TimedeltaIndex
        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        elif self.model_name == "Timer":
            setattr(self.config, "label_len", self.config.seq_len-self.config.pred_len)
        else:
            setattr(self.config, "label_len", self.config.horizon)

    def single_forecasting_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num

        if self.model_name == "MICN":
            setattr(self.config, "label_len", self.config.seq_len)
        elif self.model_name == "Timer":
            # 如果seq_len存在，则直接取，否则计算
            if hasattr(self.config, "label_len"):
                setattr(self.config, "label_len", self.config.label_len)
            else:
                setattr(self.config, "label_len", self.config.seq_len-self.config.pred_len)
        else:
            setattr(self.config, "label_len", self.config.horizon)

        # setattr(self.config, "label_len", self.config.horizon)
        

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        freq = pd.infer_freq(train_data.index)
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]
        # padding_zero = [0] * (self.config.horizon + 1)
        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def padding_data_for_forecast_np(self, test: np.ndarray) -> np.ndarray:
        """
        Padding data for prediction.

        :param test: np.ndarray, shape (l, c, n)
            A batch of time series data.
        :return: np.ndarray, shape (l + horizon, c, n)
            The padded time series data with zeros.
        """
        l, c, n = test.shape
        # 创建 padding，大小是 (horizon, c, n)
        padding = np.zeros((self.config.horizon, c, n), dtype=test.dtype)
        # 拼接在时间维度 (axis=0)
        padded = np.concatenate([test, padding], axis=0)
        return padded

    def _padding_time_stamp_mark(
            self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq,
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)

        model_name = getattr(self.config, "model_name", "") or getattr(
            self, "model_name", ""
        )
        time_encoded_models = ["STDN", "STWave", "STID"]

        if model_name in time_encoded_models:
            B, T = whole_time_stamp.shape
            flat_time = pd.to_datetime(whole_time_stamp.flatten())
            dayofweek = flat_time.weekday.values.reshape(B, T, 1)

            current_time_slice = int(getattr(self.config, "time_slice_size", 0) or 0)
            if current_time_slice <= 0:
                if T > 1:
                    sample_times = pd.to_datetime(whole_time_stamp[0, :2])
                    delta_seconds = (sample_times[1] - sample_times[0]).total_seconds()
                    current_time_slice = int(delta_seconds // 60)
                    if current_time_slice == 0:
                        current_time_slice = 1
                else:
                    current_time_slice = 5

            timeofday = (flat_time.hour * 60 + flat_time.minute) // current_time_slice
            timeofday = timeofday.values.reshape(B, T, 1)

            if model_name in ["STDN", "STWave"]:
                return np.concatenate([dayofweek, timeofday], axis=-1).astype(np.int32)

            steps_per_day = max((24 * 60) // current_time_slice, 1)
            stid_temporal = np.concatenate(
                [timeofday.astype(np.float32) / steps_per_day, dayofweek.astype(np.float32) / 7.0],
                axis=-1,
            )
            return stid_temporal

        return padding_mark

    def _infer_series_number_from_col(self, col_name: str) -> Optional[int]:
        return infer_series_number_from_col_name(col_name)

    @staticmethod
    def _is_invalid_geo_data(geo_data: Optional[np.ndarray]) -> bool:
        """
        Check whether geo_data is missing/invalid for PatchSTG.

        Invalid cases include:
        - None
        - scalar NaN
        - ndarray with all NaN values
        - ndarray whose shape is not (2, N)
        """
        if geo_data is None:
            return True

        # Handle scalar placeholders such as np.nan
        if np.isscalar(geo_data):
            try:
                return bool(np.isnan(geo_data))
            except TypeError:
                return True

        if not isinstance(geo_data, np.ndarray):
            return True

        if geo_data.ndim != 2 or geo_data.shape[0] != 2:
            return True

        if np.isnan(geo_data).all():
            return True

        return False

    def validate(
            self, valid_data_loader: DataLoader, series_dim: int, criterion: torch.nn.Module
    ) -> float:
        """
        Validates the model performance on the provided validation dataset.
        :param valid_data_loader: A PyTorch DataLoader for the validation dataset.
        :param series_dim : The number of series data‘s dimensions.
        :param criterion : The loss function to compute the loss between model predictions and ground truth.
        :returns:The mean loss computed over the validation dataset.
        """
        config = self.config
        total_loss = []
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for input, target, input_mark, target_mark in valid_data_loader:
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                exog_future = target[:, -config.horizon:, series_dim:]
                out_loss = self._process(
                    input, target, input_mark, target_mark, exog_future
                )
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]
                if len(target.shape) == 3:
                    target = target[:, -config.horizon:, :series_dim]
                    output = output[:, -config.horizon:, :series_dim]
                elif len(target.shape) == 4:
                    target = target[:, :, -config.horizon:, :series_dim]
                    output = output[:, :, -config.horizon:, :series_dim]
                    B, N, H, C = target.shape
                    # 把前两个维度合并
                    target = target.reshape(B * N, H, C)
                    output = output.reshape(B * N, H, C)
                if (
                        config.fusion_method == "mlp"
                        or config.fusion_method == "cross_attention"
                        or config.fusion_method == "conv"
                ) and self.CovariateFusion is not None:
                    output = self.CovariateFusion(exog_future, output)
                output, target = self._post_process(output, target)
                all_loss = criterion(output, target) + additional_loss
                loss = all_loss.detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        if self.CovariateFusion is not None:
            self.CovariateFusion.train()
        return total_loss

    def forecast_fit(
            self,
            train_valid_data: pd.DataFrame,
            *,
            covariates: Optional[dict] = None,
            train_ratio_in_tv: float = 1.0,
            adj_mx: Optional[dict] = None,
            **kwargs,
    ) -> "ModelBase":
        """
        Train the model.
        :param train_valid_data: Time series data used for training and validation.
        :param covariates: Additional external variables.
        :param train_ratio_in_tv: Represents the splitting ratio of the training set validation set. If it is equal to 1, it means that the validation set is not partitioned.
        :return: The fitted model object.
        """
        # 首先检查train_valid_data中有几组这样重复的数据
        if covariates is None:
            covariates = {}
        series_num = infer_series_number(train_valid_data)

        self.config.num_nodes = series_num
        series_dim = train_valid_data.shape[-1] // series_num
        if series_num * series_dim != train_valid_data.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        # 单独筛选出其中的一列作为sample_pandas，用于确定时间周期等pandas相关的数值
        sample_train_valid_data = train_valid_data.iloc[:, :series_dim]
        # 注意，此内容需要重新进行重新的划分，考虑到我已经拥有了series_num，那么我的train_valid_data和exog_data都应该从(T*(C*N))->T*C*N
        train_valid_data = self.reshape_spatiotemporal(train_valid_data, series_num)[0]
        exog_data = covariates.get("exog", None)
        if exog_data is not None:
            exog_dim = exog_data.shape[-1] // series_num
            sample_exog_data = exog_data.iloc[:, :exog_dim]
            exog_data = self.reshape_spatiotemporal(exog_data, series_num)[0]
            train_valid_data = np.concatenate([train_valid_data, exog_data], axis=1)
            exog_dim = exog_data.shape[-2]
        else:
            exog_dim = 0

        sample_train_valid_data = pd.concat([sample_train_valid_data, sample_exog_data],
                                            axis=1) if exog_data is not None else sample_train_valid_data
        if sample_train_valid_data.shape[1] == 1:
            train_drop_last = False
            self.single_forecasting_hyper_param_tune(sample_train_valid_data)
        else:
            train_drop_last = True
            self.multi_forecasting_hyper_param_tune(sample_train_valid_data)

        self.config.series_dim = series_dim
        self.config.input_dim = series_dim + exog_dim
        self.config.output_dim = series_dim
        self.config.adj_mx = adj_mx
        self.config.series_num = series_num
        geo_data = kwargs.get("geo_data", None)
        # PatchSTG requires valid geo coordinates. If absent/NaN, fail fast.
        if self.model_name == "PatchSTG":
            if self._is_invalid_geo_data(geo_data):
                raise ValueError("此数据集不支持PatchSTG：缺少有效的geo经纬度信息。")
            self.config.geo_data = geo_data
        else:
            # Keep backward compatibility for other models.
            self.config.geo_data = geo_data

        # 根据模型名，确定参数时间间隔：
        if self.model_name in ["STDN", "STWave", "STSSDL", "STID", "PatchSTG"]:
            df_stamp_raw = sample_train_valid_data.reset_index()
            # 确保提取出 datetime 对象，截图显示列名为 "date"
            raw_dates = pd.to_datetime(df_stamp_raw["date"])

            # 2. 🟢 自动化计算 STDN 专用的 time_slice_size (分钟步长)
            # 通过计算连续样本的差值来自动获取频率
            if len(raw_dates) > 1:
                # 使用 mode() 处理可能存在的时间空缺，确保步长稳健
                diffs = raw_dates.diff().dt.total_seconds()
                self.config.time_slice_size = int(diffs.mode()[0] // 60)
                if self.config.time_slice_size == 0: self.config.time_slice_size = 1
                if self.model_name == "PatchSTG":
                    self.config.tod = 24 * 60 // self.config.time_slice_size

        # 计算出相关的图的系列指标：
        if self.model_name in ["STWave"]:
            series_name = str(kwargs.get("series_name", "unknown_series"))
            dataset_stem = os.path.splitext(os.path.basename(series_name))[0]
            safe_stem = "".join(
                ch if ch.isalnum() or ch in "._-" else "_"
                for ch in dataset_stem
            )
            default_tem_graph_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "dataset",
                "tem_graph",
            )
            tem_graph_dir = os.environ.get("STWAVE_TEM_GRAPH_DIR", default_tem_graph_dir)
            tem_graph = os.path.join(
                tem_graph_dir, f"stwave_{safe_stem}_N{self.config.num_nodes}.npy"
            )
            train_data, valid_data = train_val_split(
                train_valid_data, train_ratio_in_tv, self.config.seq_len
            )
            self.config.localadj, self.config.spawave, self.config.temwave = loadGraph(
                self.config.adj_mx,
                tem_graph,
                self.config.heads * self.config.dims,
                train_data,
                sample_train_valid_data,
            )

        # 计算出有关STSSDL的图的系列指标：
        if self.model_name in ["STSSDL"]:
            from ts_benchmark.baselines.st_model.utils.STSSDL_utils import load_adj
            adj_mx = load_adj(self.config.adj_mx, self.config.adj_type)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            adj_mx = [torch.FloatTensor(i).to(device) for i in adj_mx]
            self.config.adj_mx = adj_mx
        criterion = self._init_criterion()
        self.model = self._init_model()
        if self.config.fusion_method == "mlp":
            self.CovariateFusion = MLP(self.config)
        elif self.config.fusion_method == "cross_attention":
            self.CovariateFusion = CrossAttention(self.config)
        elif self.config.fusion_method == "conv":
            self.CovariateFusion = Conv(self.config)
        else:
            self.CovariateFusion = None
        device_ids = np.arange(torch.cuda.device_count()).tolist()
        if len(device_ids) > 1 and self.config.parallel_strategy == "DP":
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            if self.CovariateFusion is not None:
                self.CovariateFusion = nn.DataParallel(
                    self.CovariateFusion, device_ids=device_ids
                )
        print(
            "----------------------------------------------------------",
            self.model_name,
        )
        config = self.config
        train_data, valid_data = train_val_split(
            train_valid_data, train_ratio_in_tv, config.seq_len
        )
        # 用于获取timestamp数据
        sample_train_data, sample_valid_data = train_val_split(
            sample_train_valid_data, train_ratio_in_tv, config.seq_len
        )
        train_data_l = train_data.shape[0]
        valid_data_l = valid_data.shape[0] if valid_data is not None else 0

        scaler_mode = self._external_scaler_mode()

        # 分别 fit 两个 scaler
        if exog_dim > 0:
            train_series_data = train_data[:, :series_dim, :]
            train_exog_data = train_data[:, series_dim:, :]
            if scaler_mode == "node_variable":
                fit_spatial_external_scaler(
                    self.scaler1,
                    train_series_data,
                    mode=scaler_mode,
                )
                if self.model_name != "STSSDL":
                    fit_spatial_external_scaler(
                        self.scaler2,
                        train_exog_data,
                        mode=scaler_mode,
                    )
            else:
                # Fit scaler1 for series data
                self.scaler1.fit(rearrange(train_series_data, "l c n -> (l n) c"))
                # Fit scaler2 for exog data
                if self.model_name != "STSSDL":
                    self.scaler2.fit(rearrange(train_exog_data, "l c n -> (l n) c"))

            # self.scaler.fit(train_data.values)
            if config.norm:
                if scaler_mode == "node_variable":
                    train_series = transform_tcn(
                        train_series_data,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                    )
                    if self.model_name == "STSSDL":
                        train_exog = transform_stssdl_exog_tcn(
                            train_exog_data,
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                    else:
                        train_exog = transform_tcn(
                            train_exog_data,
                            self.scaler2,
                            norm=True,
                            mode=scaler_mode,
                        )
                else:
                    scaled_series = self.scaler1.transform(
                        rearrange(train_series_data, "l c n -> (l n) c")
                    )
                    train_series = rearrange(scaled_series, "(l n) c -> l c n", l=train_data_l)

                    if self.model_name == "STSSDL":
                        train_exog = transform_stssdl_exog_tcn(
                            train_exog_data,
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                    else:
                        scaled_exog = self.scaler2.transform(
                            rearrange(train_exog_data, "l c n -> (l n) c")
                        )
                        train_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=train_data_l)

                train_data = np.concatenate((train_series, train_exog), axis=1)
                """
                train_data = pd.DataFrame(
                    # self.scaler.transform(train_data.values),
                    final_train_data,
                    columns=train_data.columns,
                    index=train_data.index,
                )
                """
        else:
            # Only series data, use scaler1
            if scaler_mode == "node_variable":
                fit_spatial_external_scaler(
                    self.scaler1,
                    train_data,
                    mode=scaler_mode,
                )
            else:
                self.scaler1.fit(rearrange(train_data, "l c n -> (l n) c"))
            if config.norm:
                if scaler_mode == "node_variable":
                    train_data = transform_tcn(
                        train_data,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                    )
                else:
                    scaled_data = self.scaler1.transform(
                        rearrange(train_data, "l c n -> (l n) c")
                    )
                    train_data = rearrange(scaled_data, "(l n) c -> l c n", l=train_data_l)

                """
                train_data = pd.DataFrame(
                    self.scaler1.transform(train_data.values),
                    columns=train_data.columns,
                    index=train_data.index,
                )
                """

        if train_ratio_in_tv != 1:
            if config.norm:
                if exog_dim > 0:
                    # Scale validation series data
                    if scaler_mode == "node_variable":
                        valid_series = transform_tcn(
                            valid_data[:, :series_dim, :],
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                    else:
                        scaled_series = self.scaler1.transform(
                            rearrange(valid_data[:, :series_dim, :], "l c n -> (l n) c")
                        )
                        valid_series = rearrange(scaled_series, "(l n) c -> l c n", l=valid_data_l)

                    # Scale validation exog data
                    if scaler_mode == "node_variable":
                        if self.model_name == "STSSDL":
                            valid_exog = transform_stssdl_exog_tcn(
                                valid_data[:, series_dim:, :],
                                self.scaler1,
                                norm=True,
                                mode=scaler_mode,
                            )
                        else:
                            valid_exog = transform_tcn(
                                valid_data[:, series_dim:, :],
                                self.scaler2,
                                norm=True,
                                mode=scaler_mode,
                            )
                    else:
                        if self.model_name == "STSSDL":
                            valid_exog = transform_stssdl_exog_tcn(
                                valid_data[:, series_dim:, :],
                                self.scaler1,
                                norm=True,
                                mode=scaler_mode,
                            )
                        else:
                            scaled_exog = self.scaler2.transform(
                                rearrange(valid_data[:, series_dim:, :], "l c n -> (l n) c")
                            )
                            valid_exog = rearrange(scaled_exog, "(l n) c -> l c n", l=valid_data_l)

                    # Concatenate scaled data
                    valid_data = np.concatenate(
                        (valid_series, valid_exog), axis=1
                    )

                    """
                    valid_data = pd.DataFrame(
                        final_valid_data,
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                    """
                else:
                    if scaler_mode == "node_variable":
                        valid_data = transform_tcn(
                            valid_data,
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                    else:
                        scaled_data = self.scaler1.transform(
                            rearrange(valid_data, "l c n -> (l n) c")
                        )
                        valid_data = rearrange(scaled_data, "(l n) c -> l c n", l=valid_data_l)

                    """
                    valid_data = pd.DataFrame(
                        self.scaler1.transform(valid_data.values),
                        columns=valid_data.columns,
                        index=valid_data.index,
                    )
                    """
            # 额外添加一个模型名号，然后生成即可
            valid_dataset, valid_data_loader = forecasting_data_provider(
                valid_data,
                config,
                timeenc=1,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=False,
                sample_timestamp=sample_valid_data,
                model_name=self.model_name,
            )

        train_dataset, self.train_data_loader = forecasting_data_provider(
            train_data,
            config,
            timeenc=1,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=train_drop_last,
            sample_timestamp=sample_train_data,
            model_name=self.model_name,
        )


        # Define optimizer
        optimizer = self._init_optimizer(CovariateFusion=self.CovariateFusion)
        scheduler = self._init_scheduler(optimizer, self.train_data_loader)

        if config.use_amp == 1:
            scaler = torch.cuda.amp.GradScaler()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(device)

        self.early_stopping = self._init_early_stopping()
        self.model.to(device)
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"model total trainable parameters:{total_params}")
        if self.CovariateFusion is not None:
            total_params += sum(
                p.numel() for p in self.CovariateFusion.parameters() if p.requires_grad
            )
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            if self.CovariateFusion is not None:
                self.CovariateFusion.train()
            opencity_print_train_loss = bool(
                getattr(config, "opencity_print_train_loss", False)
            )
            opencity_log_step = int(getattr(config, "opencity_log_step", 100))
            opencity_epoch_total_loss = 0.0
            opencity_step = 0
            # for input, target, input_mark, target_mark in train_data_loader:
            for i, (input, target, input_mark, target_mark) in enumerate(
                    self.train_data_loader
            ):
                optimizer.zero_grad()
                input, target, input_mark, target_mark = (
                    input.to(device),
                    target.to(device),
                    input_mark.to(device),
                    target_mark.to(device),
                )
                # decoder input
                exog_future = target[:, -config.horizon:, series_dim:].to(device)
                out_loss = self._process(
                    input, target, input_mark, target_mark, exog_future
                )
                additional_loss = 0
                output = out_loss["output"]
                if "additional_loss" in out_loss:
                    additional_loss = out_loss["additional_loss"]
                if "y_hat_l" in out_loss:
                    y_hat_l = out_loss["y_hat_l"]
                if "YL" in out_loss:
                    YL = out_loss["YL"]
                if len(output.shape) == 3:
                    target = target[:, -config.horizon:, :series_dim]
                    output = output[:, -config.horizon:, :series_dim]
                    if "y_hat_l" in out_loss and "YL" in out_loss:
                        y_hat_l = y_hat_l[:, -config.horizon:, :series_dim]
                        YL = YL[:, -config.horizon:, :series_dim]
                if len(output.shape) == 4:
                    # 此时output维度为B,N,T,C
                    target = target[:, :, -config.horizon:, :series_dim]
                    output = output[:, :, -config.horizon:, :series_dim]
                    if "y_hat_l" in out_loss and "YL" in out_loss:
                        y_hat_l = y_hat_l[:, :, -config.horizon:, :series_dim]
                        YL = YL[:, :, -config.horizon:, :series_dim]
                    B, N, H, C = target.shape

                    # 把前两个维度合并
                    target = target.reshape(B * N, H, C)
                    output = output.reshape(B * N, H, C)
                if (
                        self.config.fusion_method == "mlp"
                        or self.config.fusion_method == "conv"
                        or self.config.fusion_method == "cross_attention"
                ) and self.CovariateFusion is not None:
                    output = self.CovariateFusion(exog_future, output)
                output, target = self._post_process(output, target)

                loss = criterion(output, target)
                # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                total_loss = loss + additional_loss
                if "y_hat_l" in out_loss and "YL" in out_loss:
                    stwave_additional_loss = criterion(y_hat_l, YL)
                    total_loss += stwave_additional_loss
                # if out_loss["backcast"] is not None:
                #     backcast_loss = criterion(out_loss["backcast"], input)
                #     total_loss += backcast_loss
                backcast = out_loss.get("backcast")
                if backcast is not None:
                    backcast_loss = criterion(backcast, input)
                    total_loss += backcast_loss

                if config.use_amp == 1:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                elif self.config.lradj == "TST":
                    self._adjust_lr(optimizer, epoch + 1, config)
                if opencity_print_train_loss:
                    opencity_step += 1
                    opencity_epoch_total_loss += float(total_loss.detach().cpu())
                    if opencity_step % opencity_log_step == 0:
                        current_lr = optimizer.param_groups[0]["lr"]
                        base_loss = float(loss.detach().cpu())
                        if hasattr(additional_loss, "detach"):
                            additional_loss_value = float(additional_loss.detach().cpu())
                        else:
                            additional_loss_value = additional_loss
                        print(
                            "Benchmark OpenCity "
                            f"epoch={epoch}, "
                            f"step={opencity_step}, "
                            f"train loss is: {opencity_epoch_total_loss / opencity_step}, "
                            f"current_lr is: {current_lr}, "
                            f"base_loss={base_loss}, "
                            f"additional_loss={additional_loss_value}"
                        )

            if train_ratio_in_tv != 1:
                valid_loss = self.validate(valid_data_loader, series_dim, criterion)
                improved = self.early_stopping(valid_loss, self.model)
                if improved:
                    if self.CovariateFusion is not None:
                        self.check_point = self.save_checkpoint(
                            {
                                "Model": self.model,
                                "CovariateFusion": self.CovariateFusion,
                            }
                        )
                    else:
                        self.check_point = self.save_checkpoint({"Model": self.model})
                if self.early_stopping.early_stop:
                    break

            if scheduler is None and self.config.lradj != "TST":
                self._adjust_lr(optimizer, epoch + 1, config)

    def forecast(
            self,
            horizon: int,
            series: pd.DataFrame,
            *,
            covariates: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Make predictions.
        :param horizon: The predicted length.
        :param series: Time series data used for prediction.
        :param covariates: Additional external variables
        :return: An array of predicted results.
        """
        if covariates is None:
            covariates = {}
        series_num = infer_series_number(series)
        series_dim = series.shape[-1] // series_num
        exog_data = covariates.get("exog", None)
        if self.model_name == "Timer":
            # Timer keeps covariates in pipeline for compatibility, but does not
            # concatenate/use exogenous channels in model input path.
            exog_data = None

        # 单曲筛选出其中的一列作为sample_pandas，用于确定时间周期等pandas相关的数值
        sample_series = series.iloc[:, :series_dim]
        # 注意，此内容需要重新进行重新的划分，考虑到我已经拥有了series_num，那么我的train_valid_data和exog_data都应该从(T*(C*N))->T
        series = self.reshape_spatiotemporal(series, series_num)[0]
        
        if exog_data is not None:
            exog_dim = exog_data.shape[-1] // series_num
            sample_exog_data = exog_data.iloc[:, :exog_dim]
            sample_series = pd.concat([sample_series, sample_exog_data], axis=1)
            exog_data = self.reshape_spatiotemporal(exog_data, series_num)[0]
            # series = np.concatenate([series, exog_data], axis=1)
            exog_dim = exog_data.shape[-2]
            if exog_data.shape[-1] % series_num != 0:
                raise ValueError("Exog columns cannot be evenly divided by series_num.")
            # 生成形状为 T, C, N
            # series = np.concatenate([series, exog_data], axis=1)
            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )

        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
            if (
                    self.CovariateFusion is not None
                    and "CovariateFusion" in self.check_point
            ):
                self.CovariateFusion.load_state_dict(
                    self.check_point["CovariateFusion"]
                )

        series_data_l = series.shape[0]
        exog_data_l = exog_data.shape[0] if exog_data is not None else 0
        scaler_mode = self._external_scaler_mode()

        if self.config.norm:
            if exog_data is not None:
                if (
                        hasattr(self.config, "output_chunk_length")
                        and horizon != self.config.output_chunk_length
                ):
                    raise ValueError(
                        f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                    )
                # scaler series data with scaler1
                # series_values = series.iloc[:, :series_dim].values
                if scaler_mode == "node_variable":
                    scaled_series = transform_tcn(
                        series,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                    )
                else:
                    scaled_series = self.scaler1.transform(
                        rearrange(series, "l c n -> (l n) c", l=series_data_l)
                    )

                # scaler exog data with scaler2
                #  exog_values = series.iloc[:, series_dim:].values
                if scaler_mode == "node_variable":
                    if self.model_name == "STSSDL":
                        scaled_exog = transform_stssdl_exog_tcn(
                            exog_data,
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                    else:
                        scaled_exog = transform_tcn(
                            exog_data,
                            self.scaler2,
                            norm=True,
                            mode=scaler_mode,
                        )
                else:
                    if self.model_name == "STSSDL":
                        scaled_exog = transform_stssdl_exog_tcn(
                            exog_data,
                            self.scaler1,
                            norm=True,
                            mode=scaler_mode,
                        )
                        scaled_exog = rearrange(scaled_exog, "l c n -> (l n) c", l=exog_data_l)
                    else:
                        scaled_exog = self.scaler2.transform(
                            rearrange(exog_data, "l c n -> (l n) c", l=exog_data_l)
                        )

                # Combine scaled data
                if scaler_mode == "node_variable":
                    diff = exog_data_l - series_data_l
                    if diff > 0:
                        scaled_series = np.pad(
                            scaled_series,
                            ((0, diff), (0, 0), (0, 0)),
                            mode="constant",
                        )
                    series = np.concatenate([scaled_series, scaled_exog], axis=1)
                else:
                    diff = scaled_exog.shape[0] - scaled_series.shape[0]
                    if diff > 0:
                        scaled_series = np.pad(scaled_series, ((0, diff), (0, 0)), mode="constant")
                    scaled_values = np.concatenate([scaled_series, scaled_exog], axis=1)
                    series = rearrange(scaled_values, "(l n) c -> l c n", l=exog_data_l)
                """
                series = pd.DataFrame(
                    scaled_values,
                    columns=series.columns,
                    index=series.index,
                )
                """
            else:
                if scaler_mode == "node_variable":
                    series = transform_tcn(
                        series,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                    )
                else:
                    scaled_values = self.scaler1.transform(
                        rearrange(series, "l c n -> (l n) c")
                    )
                    series = rearrange(scaled_values, "(l n) c -> l c n", l=series_data_l)
        else:
            if exog_data is not None:
                diff = exog_data.shape[0] - series.shape[0]
                if diff > 0:
                    series = np.pad(series, ((0, diff), (0, 0), (0, 0)), mode="constant")
                series = np.concatenate([series, exog_data], axis=1)

                """
                series = pd.DataFrame(
                    self.scaler1.transform(series.values),
                    columns=series.columns,
                    index=series.index,
                )
                """

        # 存在疑问，记得咨询一下
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config
        series, test = split_time(series, len(series) - config.seq_len - horizon)
        sample_series1, sample_test = split_time(sample_series, len(sample_series) - config.seq_len - horizon)
        # test = self.padding_data_for_forecast_np(test)
        # sample_test = self.padding_data_for_forecast(sample_test)

        test_data_set, test_data_loader = forecasting_data_provider(
            test, config, timeenc=1, batch_size=1, shuffle=False, drop_last=False,
            sample_timestamp=sample_test,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
            self.CovariateFusion.eval()
        with torch.no_grad():
            answer = None
            while answer is None or answer.shape[0] < horizon:
                for input, target, input_mark, target_mark in test_data_loader:
                    input, target, input_mark, target_mark = (
                        input.to(device),
                        target.to(device),
                        input_mark.to(device),
                        target_mark.to(device),
                    )
                    # target的维度也是B，N,T,C对于上述内容理应进行重新的拼接和判断
                    if len(target.shape) == 3:
                        exog_future = target[:, -config.horizon:, series_dim:]
                    elif len(target.shape) == 4:
                        exog_future = target[:, :, -config.horizon:, series_dim:]
                    out_loss = self._process(
                        input, target, input_mark, target_mark, exog_future
                    )
                    output = out_loss["output"]
                    # 此时的output维度是(n, t, c)
                    # output的维度是B,N,T,C，也就是1,N,T,特征，我们在融合的时候需要将B*N*T*C，引入未来的外生变量
                    if self.CovariateFusion is not None:
                        if len(output.shape) == 3:
                            output = output[:, -config.horizon:, :series_dim]
                        elif len(output.shape) == 4:
                            output = output[:, :, -config.horizon:, :series_dim]
                        output = self.CovariateFusion(exog_future, output)
                        break
                    else:
                        if len(output.shape) == 3:
                            output = output[:, -config.horizon:, :series_dim]
                        elif len(output.shape) == 4:
                            output = output[:, :, -config.horizon:, :series_dim]
                        break

                column_num = output.shape[-1]
                # 再次转换维度，确保维度对应,output 尚未转化的时候，需要优先选取他的后horizon
                if len(output.shape) == 4:
                    # 此时output维度为B,N,T,C
                    B, N, H, C = output.shape
                    # 把前两个维度合并
                    output = output.reshape(B * N, H, C)
                temp1 = output[:, -config.horizon:, :]
                temp2 = rearrange(temp1, "n t c -> t (n c)", n=series_num)
                # 生成了t （nc） 的numpy数据用于拼接和归一化
                temp = temp2.cpu().numpy()
                # output= rearrange(output, "n t c -> t (n c)", n=series_num)
                # temp = output.cpu().numpy().reshape(-1, column_num)[-config.horizon :]

                if answer is None:
                    answer = temp
                else:
                    answer = np.concatenate([answer, temp], axis=0)

                if answer.shape[0] >= horizon:
                    use_prefix_slice = bool(getattr(self.config, "prefix_horizon_slice", False))

                    if self.config.norm:
                        if use_prefix_slice:
                            if scaler_mode == "node_variable":
                                answer[:horizon] = inverse_transform_wide_2d(
                                    answer[:horizon],
                                    self.scaler1,
                                    norm=True,
                                    mode=scaler_mode,
                                    series_num=series_num,
                                    series_dim=series_dim,
                                )
                            else:
                                answer_slice = rearrange(
                                    answer[:horizon], "t (n c) -> (t n) c", n=series_num
                                )
                                answer_slice = self.scaler1.inverse_transform(answer_slice)
                                answer[:horizon] = rearrange(
                                    answer_slice, "(t n) c -> t (n c)", n=series_num
                                )
                        else:
                            if scaler_mode == "node_variable":
                                answer[-horizon:] = inverse_transform_wide_2d(
                                    answer[-horizon:],
                                    self.scaler1,
                                    norm=True,
                                    mode=scaler_mode,
                                    series_num=series_num,
                                    series_dim=series_dim,
                                )
                            else:
                                answer_slice = rearrange(
                                    answer[-horizon:], "t (n c) -> (t n) c", n=series_num
                                )
                                answer_slice = self.scaler1.inverse_transform(answer_slice)
                                answer[-horizon:] = rearrange(
                                    answer_slice, "(t n) c -> t (n c)", n=series_num
                                )

                    if use_prefix_slice:
                        answer = answer[:horizon, :]
                    else:
                        answer = answer[-horizon:, :]
                    return answer

                # 此时output的维度是n,t,c
                output = output.cpu().numpy()[:, -config.horizon:]
                output1 = rearrange(output, "n t c -> t c n")
                # 填写pandas的也需要填写相应的numpy

                # for i in range(config.horizon):
                #    test.iloc[i + config.seq_len, :series_dim] = output[0, i, :]
                """
                for i in range(config.horizon):
                    test[i + config.seq_len, :series_dim,:] = output1[i, :,:]
                """
                for i in range(config.horizon):  # 时间维度
                    for j in range(series_dim):  # 通道维度
                        test[i + config.seq_len, j, :] = output1[i, j, :]

                # 填写sample_test
                for i in range(config.horizon):
                    sample_test.iloc[i + config.seq_len, :series_dim] = output[0, i, :]
                # test = test.iloc[config.horizon :]
                test = test[config.horizon:]
                sample_test = sample_test.iloc[config.horizon:]
                test = self.padding_data_for_forecast_np(test)
                sample_test = self.padding_data_for_forecast(sample_test)

                test_data_set, test_data_loader = forecasting_data_provider(
                    test,
                    config,
                    timeenc=1,
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                    sample_timestamp=sample_test
                )

    def batch_forecast(
            self, horizon: int, batch_maker: BatchMaker, exog_futures, i, series_number, **kwargs
    ) -> np.ndarray:
        """
        Make predictions by batch.

        :param horizon: The length of each prediction.
        :param batch_maker: Make batch data used for prediction.
        :param exog_futures: Future exogenous data used for prediction.
        :i: The index of the batch.
        :return: An array of predicted results.
        """
        if self.check_point is not None:
            self.model.load_state_dict(self.check_point["Model"])
            if self.CovariateFusion is not None:
                self.CovariateFusion.load_state_dict(
                    self.check_point["CovariateFusion"]
                )
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        if self.CovariateFusion is not None:
            self.CovariateFusion.to(device)
            self.CovariateFusion.eval()

        input_data = batch_maker.make_batch(self.config.batch_size, self.config.seq_len)
        input_np = input_data["input"]
        series_dim = input_np.shape[-1]
        batch_size = self.config.batch_size
        if input_data["covariates"] is None:
            covariates = {}
        else:
            covariates = input_data["covariates"]
        exog_data = covariates.get("exog")

        original_series_number = series_number
        series_dim = series_dim // series_number
        # use_timer_ci = (
        #     self.model_name == "Timer"
        #     and bool(getattr(self.config, "channel_independence", False))
        #     and original_series_dim > 1
        # )

        # if use_timer_ci:
        #     # CI mode for Timer: flatten grouped channels (N,C) -> (N*C,1)
        #     # so scaler/model dimensionality matches training (single channel).
        #     series_number = original_series_number * original_series_dim
        #     series_dim = 1

        if self.model_name == "Timer":
            # Timer keeps covariates passed through the framework, but does not
            # concatenate/use exogenous channels in rolling inference.
            exog_data = None
            exog_futures = None
        # 划分转为numpy并拼接
        # if not use_timer_ci:
        #     series_dim = series_dim // series_number
        if series_number * series_dim != input_np.shape[-1]:
            raise ValueError("Data columns cannot be evenly divided by series_num.")

        input_np = rearrange(input_np, "b t (n c) -> b n t c", n=series_number)
        if exog_data is not None:
            exog_dim = exog_data.shape[-1]
            exog_dim = exog_dim // series_number
            if series_number * exog_dim != exog_data.shape[-1]:
                raise ValueError("Data columns cannot be evenly divided by series_num.")
            exog_data = rearrange(exog_data, "b t (n c) -> b n t c", n=series_number)

            input_np = np.concatenate((input_np, exog_data), axis=3)

            if (
                    hasattr(self.config, "output_chunk_length")
                    and horizon != self.config.output_chunk_length
            ):
                raise ValueError(
                    f"Error: 'exog' is enabled during training, but horizon ({horizon}) != output_chunk_length ({self.config.output_chunk_length}) during forecast."
                )
        else:
            exog_dim = 0

        input_np_b, input_np_n, input_np_t, input_np_c = input_np.shape
        input_np = rearrange(input_np, "b n t c -> (b n) t c")
        input_np_b_n = input_np.shape[0]
        scaler_mode = self._external_scaler_mode()

        # 分为两种情况，如果我们的模型是时间序列模型，那么n和b需要合并，但是如果我们的模型是时空序列模型，则还是四维，但是注意这一步，需要在经过归一化之后再展开
        if self.config.norm:
            if exog_dim > 0:
                # Scale series data with scaler1
                series_data = input_np[:, :, :series_dim]
                # origin_shape1 = series_data.shape
                if scaler_mode == "node_variable":
                    scaled_series = transform_flat_bntc(
                        series_data,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                        series_num=original_series_number,
                    )
                else:
                    scaled_series = self.scaler1.transform(
                        rearrange(series_data, "b t c -> (b t) c")
                    )
                    scaled_series = rearrange(
                        scaled_series, "(b t) c -> b t c", b=input_np_b_n
                    )
                """

                flattened_data = series_data.reshape((-1, series_data.shape[-1]))
                series_data = self.scaler1.transform(flattened_data).reshape(
                    origin_shape1
                )
                """
                # Scale exog data with scaler2
                exog_data = input_np[:, :, series_dim:]
                if self.model_name == "STSSDL":
                    scaled_exog = transform_stssdl_exog_flat_bntc(
                        exog_data,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                        series_num=original_series_number,
                    )
                elif scaler_mode == "node_variable":
                    scaled_exog = transform_flat_bntc(
                        exog_data,
                        self.scaler2,
                        norm=True,
                        mode=scaler_mode,
                        series_num=original_series_number,
                    )
                else:
                    scaled_exog = self.scaler2.transform(
                        rearrange(exog_data, "b t c -> (b t) c")
                    )
                    scaled_exog = rearrange(
                        scaled_exog, "(b t) c -> b t c", b=input_np_b_n
                    )

                # Combine scaled data

                """
                origin_shape2 = exog_data.shape
                flattened_data = exog_data.reshape((-1, exog_data.shape[-1]))
                exog_data = self.scaler2.transform(flattened_data).reshape(
                    origin_shape2
                )
                """
                input_np = np.concatenate((scaled_series, scaled_exog), axis=2)
            else:
                """
                origin_shape = input_np.shape
                flattened_data = input_np.reshape((-1, input_np.shape[-1]))
                input_np = self.scaler1.transform(flattened_data).reshape(origin_shape)
                """
                if scaler_mode == "node_variable":
                    input_np = transform_flat_bntc(
                        input_np,
                        self.scaler1,
                        norm=True,
                        mode=scaler_mode,
                        series_num=original_series_number,
                    )
                else:
                    scaled_data = self.scaler1.transform(
                        rearrange(input_np, "b t c -> (b t) c")
                    )
                    input_np = rearrange(scaled_data, "(b t) c -> b t c", b=input_np_b_n)

        # 传入exog_futures,每个batch中对应的未来协变量
        if exog_futures is not None:
            exog_future = torch.tensor(
                exog_futures[i * batch_size: (i + 1) * batch_size, -horizon:, :]
            ).to(device)
            # exog_future 应该是 b,t,(n*c)
            if series_number * exog_dim != exog_future.shape[-1]:
                raise ValueError("Data columns cannot be evenly divided by series_num.")
            exog_future = rearrange(exog_future, "b t (n c) -> b n t c", n=series_number)
            exog_future = rearrange(exog_future, "b n t c -> (b n) t c")
            # exog_future = exog_future.float()
        else:
            exog_future = None
        if self.config.norm and exog_dim > 0:
            exog_future_np = exog_future.detach().cpu().numpy()
            if self.model_name == "STSSDL":
                exog_future = transform_stssdl_exog_flat_bntc(
                    exog_future_np,
                    self.scaler1,
                    norm=True,
                    mode=scaler_mode,
                    series_num=original_series_number,
                )
            elif scaler_mode == "node_variable":
                exog_future = transform_flat_bntc(
                    exog_future_np,
                    self.scaler2,
                    norm=True,
                    mode=scaler_mode,
                    series_num=original_series_number,
                )
            else:
                flattened_data = exog_future.reshape((-1, exog_future.shape[-1]))
                flattened_data_np = flattened_data.cpu().numpy()
                exog_future = self.scaler2.transform(flattened_data_np).reshape(
                    exog_future.shape
                )
            exog_future = torch.tensor(exog_future, dtype=torch.float32).to(device)
        input_index = input_data["time_stamps"]
        padding_len = (
                              math.ceil(horizon / self.config.horizon) + 1
                      ) * self.config.horizon
        all_mark = self._padding_time_stamp_mark(input_index, padding_len)
        all_mark = np.expand_dims(all_mark, axis=1)  # [B, 1, T, C]
        all_mark = np.repeat(all_mark, series_number, axis=1)  # [B, N, T, C]
        all_mark = rearrange(all_mark, "b n t c -> (b n) t c")
        if self.config.is_spatial == False:
            answers = self._perform_rolling_predictions(
                horizon, input_np, exog_future, series_dim, all_mark, device, 1
            )
        else:
            answers = self._perform_rolling_predictions(
                horizon, input_np, exog_future, series_dim, all_mark, device, series_number
            )

        answers = answers[:, -horizon:, :series_dim]
        if self.config.norm:
            if scaler_mode == "node_variable":
                answers = inverse_transform_flat_bntc(
                    answers,
                    self.scaler1,
                    norm=True,
                    mode=scaler_mode,
                    series_num=original_series_number,
                )
            else:
                flattened_data = answers.reshape((-1, answers.shape[-1]))
                answers = self.scaler1.inverse_transform(flattened_data).reshape(
                    answers.shape
                )
            # 可能额外需要一个四维通道

        # if use_timer_ci:
        #     # Restore grouped layout expected by evaluator target reshape:
        #     # (b * (n*c), t, 1) -> (b * n, t, c)
        #     bsz = input_data["input"].shape[0]
        #     answers = rearrange(
        #         answers,
        #         "(b n c) t 1 -> (b n) t c",
        #         b=bsz,
        #         n=original_series_number,
        #         c=original_series_dim,
        #     )
        #     series_dim = original_series_dim

        # if self.config.is_spatial == True:
        #    answers = rearrange(answers, "(b n) t c -> b n t c", n=series_number)
        # 似乎并不合理，还是应该根据维度来划分，如果是b,n,t,c任务，那么输出还是b,n,t,c，在metrics处需要额外的考虑，否则是不需要的，或需要重新思考一下
        # 完全利用if-else开辟一个，他输入的是多少，我就返回多少的情况，理论上是可行的，但是我觉得需要先放一放
        # 至少我们已经跑通了使用4维的工作流，接下来的任务就说batch的可视化分析，如何把那个batch放出来
        # 或许forecast也要修改？显示的多一个维度
        return answers[..., :series_dim]

    def _perform_rolling_predictions(
            self,
            horizon: int,
            input_np: np.ndarray,
            exog_future: torch.Tensor,
            series_dim: int,
            all_mark: np.ndarray,
            device: torch.device,
            series_number: int,
    ) -> list:
        """
        Perform rolling predictions using the given input data and marks.

        :param horizon: Length of predictions to be made.
        :param input_np: Numpy array of input data.
        :param exog_future: Future exogenous data used for prediction.
        :param series_dim: Dimension of the series data.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param device: Device to run the model on.
        :return: List of predicted results for each prediction batch.
        """
        rolling_time = 0
        input_np, target_np, input_mark_np, target_mark_np = self._get_rolling_data(
            input_np, None, all_mark, rolling_time
        )
        if exog_future is not None:
            rolling_time_sum = horizon // self.config.horizon + 1
            need_horizon = rolling_time_sum * self.config.horizon - horizon
            exog_future = torch.cat(
                (
                    exog_future,
                    torch.zeros(
                        (exog_future.shape[0], need_horizon, exog_future.shape[-1])
                    ).to(device),
                ),
                dim=1,
            )
            exog_future = exog_future.float()
        with torch.no_grad():
            answers = []
            while not answers or sum(a.shape[1] for a in answers) < horizon:
                input, dec_input, input_mark, target_mark = (
                    torch.tensor(input_np, dtype=torch.float32).to(device),
                    torch.tensor(target_np, dtype=torch.float32).to(device),
                    torch.tensor(input_mark_np, dtype=torch.float32).to(device),
                    torch.tensor(target_mark_np, dtype=torch.float32).to(device),
                )
                if exog_future is not None:
                    if len(exog_future.shape) == 3:
                        exog_future1 = exog_future[
                                       :,
                                       rolling_time
                                       * self.config.horizon: (rolling_time + 1)
                                                              * self.config.horizon,
                                       :,
                                       ]
                    else:
                        exog_future1 = exog_future[
                                       :,
                                       :,
                                       rolling_time
                                       * self.config.horizon: (rolling_time + 1)
                                                              * self.config.horizon,
                                       :,
                                       ]
                else:
                    exog_future1 = None
                if self.config.is_spatial == True:
                    input = rearrange(
                        input, "(b n) t c -> b n t c", n=series_number
                    )
                    dec_input = rearrange(
                        dec_input, "(b n) t c -> b n t c", n=series_number
                    )
                    input_mark = rearrange(
                        input_mark, "(b n) t c -> b n t c", n=series_number
                    )
                    target_mark = rearrange(
                        target_mark, "(b n) t c -> b n t c", n=series_number
                    )
                    if exog_future1 is not None:
                        exog_future1 = rearrange(
                            exog_future1, "(b n) t c -> b n t c", n=series_number
                        )
                out_loss = self._process(
                    input, dec_input, input_mark, target_mark, exog_future1
                )
                output = out_loss["output"]
                if len(output.shape) == 4:
                    # 此时output维度为B,N,T,C
                    B, N, H, C = output.shape
                    # 把前两个维度合并
                    output = output.reshape(B * N, H, C)
                if self.CovariateFusion is not None and exog_future is not None:
                    output1 = output[:, -self.config.horizon:, :series_dim]
                    output1 = self.CovariateFusion(exog_future1, output1)
                else:
                    output1 = output[:, -self.config.horizon:, :series_dim]
                column_num = output.shape[-1]
                real_batch_size = output.shape[0]
                output = torch.cat(
                    [output1, output[:, -self.config.horizon:, series_dim:]], dim=-1
                )
                answer = (
                    output.cpu()
                    .numpy()
                    .reshape(real_batch_size, -1, column_num)[
                    :, -self.config.horizon:, :
                    ]
                )
                answers.append(answer)
                if sum(a.shape[1] for a in answers) >= horizon:
                    break
                rolling_time += 1
                output = output.cpu().numpy()[:, -self.config.horizon:, :]
                (
                    input_np,
                    target_np,
                    input_mark_np,
                    target_mark_np,
                ) = self._get_rolling_data(input_np, output, all_mark, rolling_time)

        answers = np.concatenate(answers, axis=-2)
        if bool(getattr(self.config, "prefix_horizon_slice", False)):
            return answers[:, :horizon, :]
        return answers[:, -horizon:, :]

    def _get_rolling_data(
            self,
            input_np: np.ndarray,
            output: Optional[np.ndarray],
            all_mark: np.ndarray,
            rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare rolling data based on the current rolling time.

        :param input_np: Current input data.
        :param output: Output from the model prediction.
        :param all_mark: Numpy array of all marks (time stamps mark).
        :param rolling_time: Current rolling time step.
        :return: Updated input data, target data, input marks, and target marks for rolling prediction.
        """
        if rolling_time > 0:
            input_np = np.concatenate((input_np, output), axis=1)
            input_np = input_np[:, -self.config.seq_len:, :]
        target_np = np.zeros(
            (
                input_np.shape[0],
                self.config.label_len + self.config.horizon,
                input_np.shape[2],
            )
        )
        target_np[:, : self.config.label_len, :] = input_np[
                                                   :, -self.config.label_len:, :
                                                   ]
        advance_len = rolling_time * self.config.horizon
        input_mark_np = all_mark[:, advance_len: self.config.seq_len + advance_len, :]
        start = self.config.seq_len - self.config.label_len + advance_len
        end = self.config.seq_len + self.config.horizon + advance_len
        target_mark_np = all_mark[
                         :,
                         start:end,
                         :,
                         ]
        return input_np, target_np, input_mark_np, target_mark_np
