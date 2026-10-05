import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from ts_benchmark.baselines.st_model.layers.AVMDCRNN import AVWDCRNN



class AGCRN(nn.Module):
    def __init__(self, config):
        super(AGCRN, self).__init__()

        self.num_nodes = config.num_nodes
        self.feature_dim = config.enc_in
        self.input_window = config.seq_len
        self.output_window = config.pred_len
        self.output_dim = config.output_dim
        # self.hidden_dim = config.get('rnn_units', 64)
        self.hidden_dim = config.rnn_units
        # self.embed_dim = config.get('embed_dim', 10)
        self.embed_dim = config.embed_dim

        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.encoder = AVWDCRNN(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        # 事先写死等待后续调整，看看device是否要传入
        # self.device = torch.device('cuda:0')
        # self._logger = getLogger()
        # self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, input, input_mark, dec_input, target_mark):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        input = input.permute(0, 2, 1, 3)  # B, N, T, D-> B, T, N, D
        init_state = self.encoder.init_hidden(input.shape[0])
        output, _ = self.encoder(input, init_state, self.node_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]                                       # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)                           # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                      # B, T, N, C
        # B, T, N, C -> B, N, T, C
        output = output.permute(0, 2, 1, 3)
        return output
