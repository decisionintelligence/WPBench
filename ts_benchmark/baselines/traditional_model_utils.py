from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.external_scaler import (
    inverse_transform_wide_2d,
    resolve_external_scaler_mode,
    transform_wide_2d,
)
from ts_benchmark.utils.data_processing import infer_series_number


NON_DARTS_CONFIG_KEYS = {
    "norm",
    "scaler_mode",
    "external_scaler_mode",
    "internal_scaler_mode",
    "series_number",
    "series_dim",
    "target_channel",
    "target_column",
    "target_idx",
    "target_columns",
}


@dataclass
class ModelingGroup:
    input_positions: List[int]
    target_positions: List[int]
    output_columns: List[Any]


@dataclass
class TraditionalLayout:
    kind: str
    series_number: int
    series_dim: int
    target_dim: int
    full_columns: List[Any]
    output_columns: List[Any]
    groups: List[ModelingGroup]


class WideTableScaler:
    def __init__(
        self,
        *,
        norm: bool,
        mode: str,
        series_number: int,
        series_dim: int,
    ):
        self.norm = bool(norm)
        self.mode = resolve_external_scaler_mode(mode)
        self.series_number = int(series_number)
        self.series_dim = int(series_dim)
        self.scaler = StandardScaler()

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.norm:
            return frame.copy()
        self.scaler.fit(
            self._transform_input_for_fit(frame.values)
        )
        values = transform_wide_2d(
            frame.values,
            self.scaler,
            norm=True,
            mode=self.mode,
            series_num=self.series_number,
            series_dim=self.series_dim,
        )
        return pd.DataFrame(values, columns=frame.columns, index=frame.index)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.norm:
            return frame.copy()
        values = transform_wide_2d(
            frame.values,
            self.scaler,
            norm=True,
            mode=self.mode,
            series_num=self.series_number,
            series_dim=self.series_dim,
        )
        return pd.DataFrame(values, columns=frame.columns, index=frame.index)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if not self.norm:
            return values
        return inverse_transform_wide_2d(
            values,
            self.scaler,
            norm=True,
            mode=self.mode,
            series_num=self.series_number,
            series_dim=self.series_dim,
        )

    def _transform_input_for_fit(self, values: np.ndarray) -> np.ndarray:
        # t*n 状态
        if self.mode == "node_variable":
            return values
        steps = values.shape[0]
        return values.reshape(steps, self.series_number, self.series_dim).reshape(
            steps * self.series_number, self.series_dim
        )


def make_traditional_config(**kwargs):
    params = dict(kwargs)
    params.setdefault("norm", True)
    params.setdefault("scaler_mode", params.get("internal_scaler_mode", "node_variable"))
    params.setdefault("external_scaler_mode", "node_variable")
    return SimpleNamespace(**params)


def get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def resolve_internal_scaler_mode(config: Any) -> str:
    return resolve_external_scaler_mode(
        get_config_value(
            config,
            "scaler_mode",
            get_config_value(config, "internal_scaler_mode", "node_variable"),
        )
    )


def strip_non_darts_config(params: dict) -> dict:
    ret = params.copy()
    for key in NON_DARTS_CONFIG_KEYS:
        ret.pop(key, None)
    return ret


def normalize_covariate_exog(
    target_data: pd.DataFrame, covariates: Optional[dict]
) -> Optional[pd.DataFrame]:
    if not covariates or covariates.get("exog") is None:
        return None
    exog = covariates["exog"]
    if not isinstance(exog, pd.DataFrame):
        exog = pd.DataFrame(exog)
    if exog.shape[1] == 0:
        return None
    if len(exog) != len(target_data):
        if target_data.index.isin(exog.index).all():
            exog = exog.loc[target_data.index]
        elif len(exog) > len(target_data):
            exog = exog.iloc[: len(target_data)]
            exog.index = target_data.index
        else:
            raise ValueError(
                "Exogenous data length must match target history length; "
                f"got exog length {len(exog)} and target length {len(target_data)}."
            )
    elif not exog.index.equals(target_data.index):
        exog = exog.copy()
        exog.index = target_data.index
    return exog


def infer_layout(
    target_data: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    config: Any = None,
    *,
    series_number: Optional[int] = None,
    target_channel: Optional[Sequence[int]] = None,
) -> TraditionalLayout:
    explicit_series_number = (
        series_number
        or get_config_value(config, "series_number")
        or infer_series_number(target_data, default=1)
    )
    explicit_series_number = max(int(explicit_series_number), 1)

    if exog is None:
        return _infer_layout_without_exog(
            target_data,
            explicit_series_number,
            config,
            target_channel,
        )
    return _infer_layout_with_exog(
        target_data,
        exog,
        explicit_series_number,
    )


def _infer_layout_without_exog(
    target_data: pd.DataFrame,
    series_number: int,
    config: Any,
    target_channel: Optional[Sequence[int]],
) -> TraditionalLayout:
    num_cols = target_data.shape[1]
    target_cols = list(target_data.columns)

    if num_cols == 1:
        # 第一个数是这个模型看哪些列 第二个是这个模型最终预测哪些目标列 目标列的列名
        groups = [ModelingGroup([0], [0], [target_cols[0]])]
        
        # kind表示任务的类型, series_number:1 风机的个数, series_dim: 每个风机只有一个变量 target_cols：每个风机只有一个目标变量; full_colums:模型输入的列名 output_columns：模型输出的列名 groups:模型的分组信息
        return TraditionalLayout(
            "single_node_univariate", 1, 1, 1, target_cols, target_cols, groups
        )

    # 多风机数据
    if series_number > 1:
        # 首先确保风机的个数一定是总列数的因子
        if num_cols % series_number != 0:
            raise ValueError(
                "Cannot infer traditional multi-node layout: target column count "
                f"{num_cols} is not divisible by series_number {series_number}. "
                "Please provide series_number, series_dim, and target_channel/target_idx."
            )
        # 表示的是一个节点有多少个变量
        per_node = num_cols // series_number

        if per_node == 1:
            groups = [
                ModelingGroup([node], [node], [target_cols[node]])
                for node in range(series_number)
            ]
            return TraditionalLayout(
                "multi_node_univariate",
                series_number,
                1,
                1,
                target_cols,
                target_cols,
                groups,
            )
        
        # 说明每个风机有变量，这或许是指的是目标变量超过1的情况
        # 推出所有的目标列索引
        target_offsets = _resolve_target_positions(
            # 只看第一个风机的变量列，这里去的是列名
            target_data.columns[:per_node],
            # 这里就算把多个索引对应到只有第一台风机重点索引
            target_channel=_first_node_target_channel(target_channel, per_node),
            config=config,
            group_size=per_node,
            default_all=False,
        )
        output_columns = []
        groups = []
        for node in range(series_number):
            base = node * per_node
            input_positions = list(range(base, base + per_node))
            target_positions = [base + offset for offset in target_offsets]
            node_target_cols = [target_cols[pos] for pos in target_positions]
            output_columns.extend(node_target_cols)
            groups.append(
                ModelingGroup(input_positions, target_positions, node_target_cols)
            )

        # 本质是计算出了所有的目标列的索引，目标列索引为止，输入索引为止
        return TraditionalLayout(
            "multi_node_multivariate",
            series_number,
            per_node,
            len(target_offsets),
            target_cols,
            output_columns,
            groups,
        )

    target_positions = _resolve_target_positions(
        target_data.columns,
        target_channel=target_channel,
        config=config,
        group_size=num_cols,
        default_all=True,
    )
    groups = [
        ModelingGroup(
            list(range(num_cols)),
            target_positions,
            [target_cols[pos] for pos in target_positions],
        )
    ]
    # 总体而言就算返回了数据集类型，风机数量，每个风机有几个变量，每个风机有几个目标变量，原始完整列名，最终需要输出的目标列名，group对应了输入列/预测列/对应的列名
    return TraditionalLayout(
        "single_node_multivariate",
        1,
        num_cols,
        len(target_positions),
        target_cols,
        [target_cols[pos] for pos in target_positions],
        groups,
    )


def _infer_layout_with_exog(
    target_data: pd.DataFrame,
    exog: pd.DataFrame,
    series_number: int,
) -> TraditionalLayout:
    target_total = target_data.shape[1]
    exog_total = exog.shape[1]
    if target_total % series_number != 0 or exog_total % series_number != 0:
        raise ValueError(
            "Cannot infer traditional multi-node multivariate layout: target/exog "
            f"columns ({target_total}/{exog_total}) are not divisible by "
            f"series_number {series_number}. Please provide series_number, "
            "series_dim, and target_channel/target_idx."
        )

    target_dim = target_total // series_number
    exog_dim = exog_total // series_number
    series_dim = target_dim + exog_dim
    target_cols = list(target_data.columns)
    exog_cols = list(exog.columns)
    full_columns = []
    output_columns = []
    groups = []

    for node in range(series_number):
        target_start = node * target_dim
        target_end = target_start + target_dim
        exog_start = node * exog_dim
        exog_end = exog_start + exog_dim

        input_positions = list(range(node * series_dim, (node + 1) * series_dim))
        target_positions = list(range(node * series_dim, node * series_dim + target_dim))
        node_target_cols = target_cols[target_start:target_end]
        node_exog_cols = exog_cols[exog_start:exog_end]
        full_columns.extend(node_target_cols + node_exog_cols)
        output_columns.extend(node_target_cols)
        groups.append(ModelingGroup(input_positions, target_positions, node_target_cols))

    kind = "single_node_multivariate" if series_number == 1 else "multi_node_multivariate"
    return TraditionalLayout(
        kind,
        series_number,
        series_dim,
        target_dim,
        full_columns,
        output_columns,
        groups,
    )


def build_full_frame(
    target_data: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    layout: TraditionalLayout,
) -> pd.DataFrame:
    if exog is None:
        return target_data.loc[:, layout.full_columns].copy()

    pieces = []
    # 每个风机的目标列
    target_dim = layout.target_dim
    # 外生变量的数量
    exog_dim = layout.series_dim - layout.target_dim
    for node in range(layout.series_number):
        target_start = node * target_dim
        exog_start = node * exog_dim
        pieces.append(target_data.iloc[:, target_start : target_start + target_dim])
        pieces.append(exog.iloc[:, exog_start : exog_start + exog_dim])
    # 构成形如：full_frame=
    # [power_0, speed_0, dir_0,
    #  power_1, speed_1, dir_1]
    return pd.concat(pieces, axis=1)


def select_target_positions(layout: TraditionalLayout) -> List[int]:
    positions = []
    for group in layout.groups:
        positions.extend(group.target_positions)
    return positions


def _resolve_target_positions(
    columns: Sequence[Any],
    *,
    target_channel: Optional[Sequence[int]],
    config: Any,
    group_size: int,
    default_all: bool,
) -> List[int]:
    configured = (
        get_config_value(config, "target_idx")
        or get_config_value(config, "target_column")
        or get_config_value(config, "target_columns")
        or target_channel
    )
    if configured is None:
        return list(range(group_size)) if default_all else [0]
    if isinstance(configured, (str, int)):
        configured = [configured]

    positions = []
    col_list = list(columns)
    for item in configured:
        if isinstance(item, str):
            if item not in col_list:
                raise ValueError(f"Configured target column {item!r} is not present.")
            positions.append(col_list.index(item))
        else:
            idx = int(item)
            if idx < 0:
                idx += group_size
            if idx < 0 or idx >= group_size:
                raise ValueError(
                    f"Configured target index {item} is out of range for group size {group_size}."
                )
            positions.append(idx)
    return list(dict.fromkeys(positions))


def _first_node_target_channel(
    target_channel: Optional[Sequence[int]], group_size: int
) -> Optional[List[int]]:
    if target_channel is None:
        return None
    if isinstance(target_channel, (str, int)):
        target_channel = [target_channel]
    first_node = []
    for item in target_channel:
        if isinstance(item, str):
            first_node.append(item)
            continue
        idx = int(item)
        if 0 <= idx < group_size:
            first_node.append(idx)
    return first_node or None
