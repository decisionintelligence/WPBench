import os

import numpy as np
import pandas as pd
import logging

FREQ_MAP = {
    "Y": "yearly",
    "A": "yearly",
    "A-DEC": "yearly",
    "A-JAN": "yearly",
    "A-FEB": "yearly",
    "A-MAR": "yearly",
    "A-APR": "yearly",
    "A-MAY": "yearly",
    "A-JUN": "yearly",
    "A-JUL": "yearly",
    "A-AUG": "yearly",
    "A-SEP": "yearly",
    "A-OCT": "yearly",
    "A-NOV": "yearly",
    "AS-DEC": "yearly",
    "AS-JAN": "yearly",
    "AS-FEB": "yearly",
    "AS-MAR": "yearly",
    "AS-APR": "yearly",
    "AS-MAY": "yearly",
    "AS-JUN": "yearly",
    "AS-JUL": "yearly",
    "AS-AUG": "yearly",
    "AS-SEP": "yearly",
    "AS-OCT": "yearly",
    "AS-NOV": "yearly",
    "BA-DEC": "yearly",
    "BA-JAN": "yearly",
    "BA-FEB": "yearly",
    "BA-MAR": "yearly",
    "BA-APR": "yearly",
    "BA-MAY": "yearly",
    "BA-JUN": "yearly",
    "BA-JUL": "yearly",
    "BA-AUG": "yearly",
    "BA-SEP": "yearly",
    "BA-OCT": "yearly",
    "BA-NOV": "yearly",
    "BAS-DEC": "yearly",
    "BAS-JAN": "yearly",
    "BAS-FEB": "yearly",
    "BAS-MAR": "yearly",
    "BAS-APR": "yearly",
    "BAS-MAY": "yearly",
    "BAS-JUN": "yearly",
    "BAS-JUL": "yearly",
    "BAS-AUG": "yearly",
    "BAS-SEP": "yearly",
    "BAS-OCT": "yearly",
    "BAS-NOV": "yearly",
    "Q": "quarterly",
    "Q-DEC": "quarterly",
    "Q-JAN": "quarterly",
    "Q-FEB": "quarterly",
    "Q-MAR": "quarterly",
    "Q-APR": "quarterly",
    "Q-MAY": "quarterly",
    "Q-JUN": "quarterly",
    "Q-JUL": "quarterly",
    "Q-AUG": "quarterly",
    "Q-SEP": "quarterly",
    "Q-OCT": "quarterly",
    "Q-NOV": "quarterly",
    "QS-DEC": "quarterly",
    "QS-JAN": "quarterly",
    "QS-FEB": "quarterly",
    "QS-MAR": "quarterly",
    "QS-APR": "quarterly",
    "QS-MAY": "quarterly",
    "QS-JUN": "quarterly",
    "QS-JUL": "quarterly",
    "QS-AUG": "quarterly",
    "QS-SEP": "quarterly",
    "QS-OCT": "quarterly",
    "QS-NOV": "quarterly",
    "BQ-DEC": "quarterly",
    "BQ-JAN": "quarterly",
    "BQ-FEB": "quarterly",
    "BQ-MAR": "quarterly",
    "BQ-APR": "quarterly",
    "BQ-MAY": "quarterly",
    "BQ-JUN": "quarterly",
    "BQ-JUL": "quarterly",
    "BQ-AUG": "quarterly",
    "BQ-SEP": "quarterly",
    "BQ-OCT": "quarterly",
    "BQ-NOV": "quarterly",
    "BQS-DEC": "quarterly",
    "BQS-JAN": "quarterly",
    "BQS-FEB": "quarterly",
    "BQS-MAR": "quarterly",
    "BQS-APR": "quarterly",
    "BQS-MAY": "quarterly",
    "BQS-JUN": "quarterly",
    "BQS-JUL": "quarterly",
    "BQS-AUG": "quarterly",
    "BQS-SEP": "quarterly",
    "BQS-OCT": "quarterly",
    "BQS-NOV": "quarterly",
    "M": "monthly",
    "BM": "monthly",
    "CBM": "monthly",
    "MS": "monthly",
    "BMS": "monthly",
    "CBMS": "monthly",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "D": "daily",
    "B": "daily",
    "C": "daily",
    "H": "hourly",
    "UNKNOWN": "other",
}


def read_data(path: str, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.
    According to the provided file path, read the data file and return the corresponding DataFrame.
    :param path: The path to the data file.
    :return:  The DataFrame of the content of the data file.
    """
    data = pd.read_csv(path)
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    # 计算每一个特征值有几个
    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def read_data_rel(path: str, data, nrows=None, weight_adj_epsilon: float = 0.5) -> np.ndarray:
    """
    Read a .rel file and convert it into a directed adjacency matrix (NumPy array)
    using the distances/costs in the file, strictly following the file's content.

    Parameters
    ----------
    path : str
        Path to the .rel file.
    nrows : int, optional
        Number of rows to read (default: None, read all).
    weight_adj_epsilon : float, optional
        Threshold for Gaussian kernel (default=0.1).

    Returns
    -------
    np.ndarray
        The adjacency matrix (num_sensors x num_sensors), ordered by IDs in the file.
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.warning(f"[read_data_rel] Relation file not found: {path}, Based on DataFrame calculate corr_matrix instead.")
        # print(data)
        # 获取所有列名
        all_cols = data.columns.tolist()
        if 'label' in all_cols:
            all_cols.remove('label')

        # 🎯 核心逻辑：识别每个 type 的第一列
        # 根据 image_1c131c.png，特征是以 'type x:特征' 命名的
        # 我们通过解析 ':' 前的字符串来对风机进行分组
        type_groups = {}
        for col in all_cols:
            t_name = col.split(':')[0] if ':' in col else col
            if t_name not in type_groups:
                type_groups[t_name] = col  # 只记录该组遇到的第一个列名

        # 提取所有组的第一列名
        representative_cols = list(type_groups.values())
        num_turbines = len(representative_cols)

        # ⚠️ 健壮性检查：如果节点数量不足以构成矩阵，则返回 None
        if num_turbines <= 1:
            logger.warning(f"检测到的风机数量为 {num_turbines}，无法构建关系矩阵。")
            return None

        logger.info(f"检测到 {num_turbines} 台风机，提取列: {representative_cols}")

        # 📈 计算皮尔逊相关系数
        data_to_corr = data[representative_cols]
        corr_matrix = data_to_corr.corr().values
        adj_mx = np.nan_to_num(corr_matrix)

        # ✂️ 阈值过滤
        adj_mx[adj_mx < weight_adj_epsilon] = 0
        adj_mx[adj_mx >= weight_adj_epsilon] = 1

        # 🔄 添加自环以支持 GCN 计算
        np.fill_diagonal(adj_mx, 1.0)
        return adj_mx

    # 1️⃣ 读取 rel 文件
    rel_df = pd.read_csv(path, nrows=nrows, dtype={"origin_id": str, "destination_id": str})

    if "type" in rel_df.columns and (rel_df["type"] == "npy_weight").all():
          sensor_ids = sorted(list(set(rel_df["origin_id"]) | set(rel_df["destination_id"])))
          N = len(sensor_ids)
          adj_mx = np.zeros((N, N), dtype=np.float32)

          for _, row in rel_df.iterrows():
              src, dst = int(row["origin_id"]), int(row["destination_id"])
              adj_mx[src, dst] = float(row["cost"])

          logger.info(f"[read_data_rel] Loaded npy_weight {path}, shape={adj_mx.shape}")
          return adj_mx

    # 2️⃣ 获取所有出现过的节点 ID
    sensor_ids = sorted(list(set(rel_df["origin_id"]) | set(rel_df["destination_id"])))
    id_to_index = {sid: i for i, sid in enumerate(sensor_ids)}
    N = len(sensor_ids)

    # 3️⃣ 初始化邻接矩阵为 inf
    adj_mx = np.full((N, N), np.inf, dtype=np.float32)

    # 4️⃣ 填充矩阵（严格按照文件，有向）
    for _, row in rel_df.iterrows():
        src, dst = int(row["origin_id"]), int(row["destination_id"])
        value = row["cost"] if "cost" in row else 1.0
        adj_mx[src, dst] = value

    # 5️⃣ 高斯核计算权重，只对存在的边
    mask = ~np.isinf(adj_mx)
    distances = adj_mx[mask].flatten()
    std = distances.std() if len(distances) > 0 else 1.0
    adj_mx[mask] = np.exp(-np.square(adj_mx[mask] / std))
    adj_mx[mask & (adj_mx < weight_adj_epsilon)] = 0
    adj_mx[~mask] = 0  # 不存在的边保持为0

    # 6️⃣ 添加自环
    np.fill_diagonal(adj_mx, 1.0)

    logger.info(f"[read_data_rel] Loaded {path}, shape={adj_mx.shape}")
    return adj_mx


def read_data_geo(path: str):
    """
    读取 .geo 文件，返回形如 (2, N) 的经纬度 numpy 数组。
    若文件不存在则返回 None（优雅降级，不影响其他模型）。

    .geo 文件格式（CSV，至少包含以下两列）：
        node_id, Lat, Lng
        0,       39.9, 116.3
        ...

    :param path: .geo 文件路径。
    :return: np.ndarray shape=(2, N)，其中 row-0=Lat, row-1=Lng；或 None。
    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(path):
        logger.warning(
            "[read_data_geo] .geo file not found: %s, returning None.", path
        )
        return None
    meta = pd.read_csv(path)
    if "Lat" not in meta.columns or "Lng" not in meta.columns:
        logger.warning(
            "[read_data_geo] .geo file %s missing 'Lat'/'Lng' columns, returning None.",
            path,
        )
        return None
    lat = meta["Lat"].values
    lng = meta["Lng"].values
    locations = np.stack([lat, lng], axis=0)  # shape: (2, N)
    logger.info("[read_data_geo] Loaded %s, %d nodes.", path, locations.shape[1])
    return locations


def load_series_info(file_path: str) -> dict:
    """
    get series info
    :param file_path: series file path
    :return: series info
    :rtype: dict
    """
    data = read_data(file_path)
    file_name = os.path.basename(file_path)
    freq = pd.infer_freq(data.index)
    freq = FREQ_MAP.get(freq, "other")
    if_univariate = data.shape[1] == 1
    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": data.shape[0],
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }
