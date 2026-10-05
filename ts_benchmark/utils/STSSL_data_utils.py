import pandas as pd
import numpy as np


def generate_stssdl_features_df(data: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
    """
    为 ST-SSDL 模型动态生成带有历史日周期锚点的 2D DataFrame 宽表。
    自动推断时间分辨率，并严格按列名规律组合特征。

    :param data: 原始 DataFrame，包含所有 type 的所有特征，索引为 DatetimeIndex。
    :param train_ratio: 训练集比例，用于框定计算均值的数据范围。
    :return: 形状为 (T, N * 3) 的 DataFrame。
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    # 1. 动态推断时间分辨率
    time_diff = data.index[1] - data.index[0]
    freq_minutes = int(time_diff.total_seconds() / 60)
    print(f"🕒 动态推断的时间分辨率为: {freq_minutes} 分钟")

    # 2. 严格提取每个 type 的第一个属性
    target_columns = []
    prefixes = []
    seen_types = set()

# 利用set，筛选出每个type排列在第一个的变量
    for col in data.columns:
        if ':' in str(col):
            type_prefix = str(col).split(':')[0].strip()  # 提取 "type 1"
            if type_prefix not in seen_types:
                target_columns.append(col)
                prefixes.append(type_prefix)
                seen_types.add(type_prefix)
        else:
            if col not in seen_types:
                target_columns.append(col)
                prefixes.append(col)
                seen_types.add(col)

    target_data = data[target_columns]

    # 3. 计算通用时间特征 timeofday (0~1)
    timeofday = (target_data.index - target_data.index.normalize()) / pd.Timedelta('1D')

    # 4. 计算历史周周期锚点，对齐原版 ST-SSDL 的 weekdaytime 语义
    train_size = int(len(target_data) * train_ratio)
    steps_per_day = max((24 * 60) // freq_minutes, 1)
    slot_in_day = (target_data.index.hour * 60 + target_data.index.minute) // freq_minutes
    weekday_slots = target_data.index.weekday * steps_per_day + slot_in_day

    train_data = target_data.iloc[:train_size].replace(0, np.nan).copy()
    train_data['weekday_slot'] = weekday_slots[:train_size]

    historical_mean = train_data.groupby('weekday_slot').mean()

    # 映射回全量时间轴，并处理可能的空缺值
    flow_y_df = historical_mean.reindex(weekday_slots).reset_index(drop=True)
    flow_y_df = flow_y_df.ffill().bfill().fillna(0)
    flow_y_df.index = target_data.index  # 重新对齐时间索引

    # ==========================================
    # 💡 核心改进：字典组装，生成横向平铺宽表
    # ==========================================
    final_dict = {}
    for i, col in enumerate(target_columns):
        prefix = prefixes[i]

        # 第 1 列：原始值 (比如 'type 1:wp')
        final_dict[col] = target_data[col].values

        # 第 2 列：时间特征 (比如 'type 1:timeofday')
        final_dict[f"{prefix}:timeofday"] = timeofday.values

        # 第 3 列：历史均值锚点 (比如 'type 1:label_Y')
        final_dict[f"{prefix}:label_Y"] = flow_y_df[col].values

    final_df = pd.DataFrame(final_dict, index=target_data.index)

    print(f"✅ 成功生成平铺 DataFrame，特征维度由 {data.shape[1]} 转变为 {final_df.shape[1]} (N*3)")
    return final_df
