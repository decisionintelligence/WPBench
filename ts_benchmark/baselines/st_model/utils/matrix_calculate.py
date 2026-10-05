import numpy as np
import scipy.sparse as sp
import torch


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # print(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num


def cal_lape(adj_mx):
    lape_dim = 32
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 提取实际存在的特征向量 (针对你的情况是 6 个)
    # 注意切片保护，防止越界
    laplacian_pe = EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]

    # 🚀 关键修复：如果特征向量不足 32 个，用 0 补齐
    if laplacian_pe.shape[1] < lape_dim:
        pad_width = lape_dim - laplacian_pe.shape[1]
        padding = np.zeros((laplacian_pe.shape[0], pad_width))
        laplacian_pe = np.hstack([laplacian_pe, padding])

    return laplacian_pe

def _calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # self._logger.info(f"Number of isolated points: {isolated_point_num}")
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def _cal_lape_pdformer(adj_mx, lape_dim=8):
    L, isolated_point_num = _calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]).float()
    laplacian_pe.require_grad = False
    return laplacian_pe