import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from ts_benchmark.baselines.st_model.layers.STDN_model import TEmbedding, SEmbedding, Trend, Seasonal, FeedForward, \
    GRUEncoder, AttentionDecoder, FC, gcn

# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")、
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class STDNModel(nn.Module):
    def __init__(self, configs, bn_decay) -> None:
        super(STDNModel, self).__init__()
        L = int(configs.L)
        K = int(configs.K)
        d = int(configs.d)
        self.L = L
        self.K = K
        self.d = d
        # self.node_miss_rate = float(configs.node_miss_rate)
        # self.T_miss_len = int(configs.T_miss_len)
        self.order = int(configs.order)
        print('L', self.L)
        print('K', self.K)
        print('d', self.d)
        D = K * d
        set_dim = int(configs.reference)
        self.num_his = int(configs.seq_len)
        time_slice_size = int(configs.time_slice_size)
        self.input_dim = int(1440 / time_slice_size) + 7
        self.num_pred = int(configs.pred_len)
        self.num_of_vertices = int(configs.series_num)
        self.TEmbedding = TEmbedding(self.input_dim, D, self.num_of_vertices, bn_decay)
        self.SEmbedding = SEmbedding(D)
        self.Trend = Trend()
        self.Seasonal = Seasonal()
        self.FeedForward_for_t = FeedForward([D, D], res_ln=True)
        self.FeedForward_for_s = FeedForward([D, D], res_ln=True)
        self.GRU_Trend = GRUEncoder(D, self.num_his)
        self.GRU_Seasonal = GRUEncoder(D, self.num_his)
        self.Decoder = nn.ModuleList(
            [AttentionDecoder(K, d, self.num_of_vertices, set_dim, bn_decay) for _ in range(L)])
        out_channels = 1

        self.FC_1 = FC(input_dims=[1, D], units=[D, D], activations=[F.relu, None],
                       bn_decay=bn_decay)  # in_channels=3
        self.FC_2 = FC(input_dims=[D, D], units=[D, out_channels], activations=[F.relu, None],
                       bn_decay=bn_decay)

        # dynamic GCN
        self.nodevec_p1 = nn.Parameter(torch.randn(int(1440 / time_slice_size), D).to(device), requires_grad=True).to(
            device)
        self.nodevec_p2 = nn.Parameter(torch.randn(int(configs.series_num), D).to(device),
                                       requires_grad=True).to(device)
        self.nodevec_p3 = nn.Parameter(torch.randn(int(configs.series_num), D).to(device),
                                       requires_grad=True).to(device)
        self.nodevec_pk = nn.Parameter(torch.randn(D, D, D).to(device), requires_grad=True).to(device)
        self.GCN = gcn(D, D, order=self.order)

    def dgconstruct(self, time_embedding, source_embedding, target_embedding, core_embedding):
        adp = torch.einsum('ai, ijk->ajk', time_embedding, core_embedding)
        adp = torch.einsum('bj, ajk->abk', source_embedding, adp)
        adp = torch.einsum('ck, abk->abc', target_embedding, adp)
        adp = F.softmax(F.relu(adp), dim=2)
        # print(adp.shape)
        return adp

    def forward(self, X, TE, lpls,):
        # input
        X = X[:,:,:,0]
        X = X.unsqueeze(-1)
        X = X.permute(0,2,1,3).contiguous()
        X = self.FC_1(X)
        ind = TE[:, 0, 1]
        ind = torch.tensor(ind, dtype=torch.long)
        # dynamic graph construction
        adp = self.dgconstruct(self.nodevec_p1[ind], self.nodevec_p2, self.nodevec_p3, self.nodevec_pk)
        new_supports = [adp]
        X = self.GCN(X, new_supports)

        SE = self.SEmbedding(lpls, X.shape[0], self.num_pred)
        his, pred = self.TEmbedding(TE, SE, self.input_dim - 7, self.num_of_vertices, self.num_his)
        trend = self.Trend(X, his)
        seasonal = self.Seasonal(X, trend)
        trend = self.FeedForward_for_t(trend)
        seasonal = self.FeedForward_for_s(seasonal)
        # encoder
        trend = self.GRU_Trend(trend)
        seasonal = self.GRU_Seasonal(seasonal)

        result = trend + seasonal
        # decoder
        for net in self.Decoder:
            result = net(result, pred, SE, None)
        result = self.FC_2(result)
        del TE, his, trend, seasonal
        return result

# def make_model( configs, bn_decay=0.1):
#     model = STDN( configs, bn_decay=0.1)
#     for p in model.parameters():
#         if p.dim() > 1:
#             nn.init.xavier_uniform_(p)
#     return model