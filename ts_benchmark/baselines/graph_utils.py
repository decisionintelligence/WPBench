import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from ts_benchmark.baselines.utils import train_val_split


def laplacian(W):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    d = 1 / np.sqrt(d)
    D = sp.diags(d, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D * W * D
    return L

def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return (lamb, U)


def get_eigv(adj, k):
    L = laplacian(adj)
    num_nodes = L.shape[0]

    # 将稀疏矩阵转为密集矩阵，使用 eigh 计算全部特征值（eigh 非常稳定，不会像 eigsh 那样在 k>N 时报错）
    if sp.issparse(L):
        L = L.toarray()

    # 计算全部特征值和特征向量
    lamb, U = np.linalg.eigh(L)

    # 🚀 核心修复：如果节点数(7) 小于 模型需要的维度(128)，则自动补 0 对齐
    if num_nodes < k:
        # 特征值补齐：(7,) -> (128,)
        lamb_padded = np.zeros(k)
        lamb_padded[:num_nodes] = lamb

        # 特征向量补齐：(7, 7) -> (7, 128)
        U_padded = np.zeros((num_nodes, k))
        U_padded[:, :num_nodes] = U

        return (lamb_padded, U_padded)
    else:
        # 如果是大图(如 307 个节点)，则按原逻辑取最大的 k 个
        return (lamb[-k:], U[:, -k:])


def construct_tem_adj(data, num_node, sample):
    # 1. 动态计算时间分辨率和每天的步数
    # 假设 sample 是一个 DataFrame 并且有时间列（或者直接传入了 Series）
    # 1. 动态计算时间分辨率和每天的步数
    if isinstance(sample, pd.DataFrame):
        # ✅ 直接把索引 (DatetimeIndex) 转换成 Series
        time_series = pd.Series(sample.index)
    else:
        time_series = pd.to_datetime(sample)

    if len(time_series) > 1:
        diffs = time_series.diff().dt.total_seconds()
        time_slice_size = int(diffs.mode()[0] // 60)
        if time_slice_size == 0: time_slice_size = 1
    else:
        time_slice_size = 5  # 兜底默认 5 分钟

    # 动态计算每天有多少个时间步 (例如 15分钟就是 96)
    steps_per_day = (24 * 60) // time_slice_size
    # 动态计算 DTW 的 radius (保持物理时间窗口约为 30 分钟)
    dtw_radius = max(1, 30 // time_slice_size)

    # print(f"Auto-detected resolution: {time_slice_size} mins. "
    #       f"Steps per day: {steps_per_day}, DTW radius: {dtw_radius}")

    # 2. 提取每个节点的典型日均波动曲线
    data_mean = np.mean([data[steps_per_day * i: steps_per_day * (i + 1)]
                         for i in range(data.shape[0] // steps_per_day)], axis=0)
    data_mean = data_mean.squeeze().T

    # 3. 计算 DTW 距离
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node), desc="Calculating DTW Matrix"):
        for j in range(i, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=dtw_radius)[0]

    for i in range(num_node):
        for j in range(i):
            dtw_distance[i][j] = dtw_distance[j][i]

    # 4. 稀疏化构造 0-1 语义矩阵
    nth = np.sort(dtw_distance.reshape(-1))[
          int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]):
          int(np.log2(dtw_distance.shape[0]) * dtw_distance.shape[0]) + 1
          ]  # NlogN edges

    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
    return tem_matrix


def loadGraph(spatial_graph, temporal_graph, dims, data, sample):
    # 1. 空间邻接矩阵准备 (假设传进来的 spatial_graph 已经是计算好的 numpy 矩阵)
    if spatial_graph is None:
        raise ValueError("STWave requires a non-empty adjacency matrix; got adj_mx=None.")
    adj = spatial_graph
    adj = adj + np.eye(adj.shape[0])  # 加上自环，防止模型过平滑丢失自身特征

    def build_temporal_graph():
        print(f"Temporal graph not found or stale. Constructing new one...")
        # 修复切片 Bug：原数据通常是 [T, N, C]，提取风速/功率的主特征 (C=0)
        # 如果你的 data 已经是 [T, N] 的二维数据，就不需要切片了
        if len(data.shape) == 3:
            data_for_dtw = data[:, 0, :]
        else:
            data_for_dtw = data

        rebuilt_tem_adj = construct_tem_adj(data_for_dtw, adj.shape[0], sample)

        # 补充缺失的父级目录
        save_dir = os.path.dirname(temporal_graph)
        if save_dir:  # 确保路径不为空字符串
            os.makedirs(save_dir, exist_ok=True)
        np.save(temporal_graph, rebuilt_tem_adj)
        return rebuilt_tem_adj

    # 2. 时间语义矩阵准备 (懒加载机制)
    if os.path.exists(temporal_graph):
        print(f"Loading cached temporal graph from {temporal_graph}...")
        tem_adj = np.load(temporal_graph)
        if tem_adj.shape[0] != adj.shape[0]:
            print(
                f"Cached temporal graph shape {tem_adj.shape} does not match "
                f"spatial graph shape {adj.shape}; rebuilding..."
            )
            tem_adj = build_temporal_graph()
    else:
        tem_adj = build_temporal_graph()

    # 3. 获取图小波位置编码
    spawave = get_eigv(adj, dims)
    temwave = get_eigv(tem_adj, dims)

    # 4. 获取局部邻居 (用于 ESGAT 的 Query Sampling)
    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix == 0] = dist_matrix.max() + 10
    localadj = np.argpartition(dist_matrix, sampled_nodes_number, -1)[:, :sampled_nodes_number]

    return localadj, spawave, temwave
