import math

import torch
import torch.nn as nn


class BigST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_nodes = getattr(config, "series_num", getattr(config, "num_nodes"))
        self.seq_len = config.seq_len
        self.output_len = getattr(config, "pred_len", getattr(config, "horizon"))
        self.input_dim = getattr(config, "enc_in", getattr(config, "input_dim", 1))
        self.output_dim = getattr(config, "output_dim", getattr(config, "series_dim", 1))
        self.hidden_dim = getattr(config, "bigst_hidden_dim", getattr(config, "hid_dim", 32))
        self.node_dim = getattr(config, "bigst_node_dim", getattr(config, "node_dim", 32))
        self.time_dim = getattr(config, "bigst_time_dim", getattr(config, "time_dim", 32))
        self.num_layers = getattr(config, "bigst_num_layers", getattr(config, "num_layers", 3))
        self.tau = getattr(config, "bigst_tau", getattr(config, "tau", 1.0))
        self.random_feature_dim = getattr(
            config, "bigst_random_feature_dim", getattr(config, "random_feature_dim", 64)
        )
        self.use_residual = getattr(config, "bigst_use_residual", getattr(config, "use_residual", True))
        self.use_bn = getattr(config, "bigst_use_bn", getattr(config, "use_bn", True))
        self.dropout = getattr(config, "dropout", 0.1)
        time_slice_size = max(int(getattr(config, "time_slice_size", 5) or 5), 1)
        self.time_num = getattr(config, "bigst_time_num", 24 * 60 // time_slice_size)
        self.week_num = getattr(config, "bigst_week_num", 7)

        self.node_emb_layer = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, self.time_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, self.time_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        self.input_emb_layer = nn.Conv2d(
            self.seq_len * (self.input_dim + 2), self.hidden_dim, kernel_size=(1, 1)
        )
        graph_dim = self.node_dim + self.time_dim * 2
        self.W_1 = nn.Conv2d(graph_dim, self.hidden_dim, kernel_size=(1, 1))
        self.W_2 = nn.Conv2d(graph_dim, self.hidden_dim, kernel_size=(1, 1))

        block_dim = self.hidden_dim + graph_dim
        self.linear_conv = nn.ModuleList(
            [
                LinearizedConv(
                    block_dim,
                    block_dim,
                    self.dropout,
                    self.tau,
                    self.random_feature_dim,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.bn = nn.ModuleList([nn.LayerNorm(block_dim) for _ in range(self.num_layers)])
        self.activation = nn.ReLU()
        self.regression_layer = nn.Conv2d(
            block_dim * 2, self.output_len * self.output_dim, kernel_size=(1, 1)
        )

    def forward(self, input, input_mark=None, dec_input=None, target_mark=None):
        history_data = self._build_history_data(input, input_mark)
        batch_size, num_nodes, _, _ = history_data.shape

        time_idx = self._time_index(history_data[..., 1])
        week_idx = self._week_index(history_data[..., 2])
        time_emb = self.time_emb_layer[time_idx[:, :, -1]]
        week_emb = self.week_emb_layer[week_idx[:, :, -1]]

        x = history_data.contiguous().view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        input_emb = self.input_emb_layer(x)
        node_emb = self.node_emb_layer.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1)
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)

        x_pool = [x]
        node_vec1 = self.W_1(x_g).permute(0, 2, 3, 1)
        node_vec2 = self.W_2(x_g).permute(0, 2, 3, 1)
        for i, conv in enumerate(self.linear_conv):
            residual = x
            x = conv(x, node_vec1, node_vec2)
            if self.use_residual:
                x = x + residual
            if self.use_bn:
                x = self.bn[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        x_pool.append(x)
        x = self.activation(torch.cat(x_pool, dim=1))
        output = self.regression_layer(x).squeeze(-1)
        output = output.view(batch_size, self.output_len, self.output_dim, num_nodes)
        return output.permute(0, 3, 1, 2).contiguous()

    def _build_history_data(self, input, input_mark):
        if input.dim() != 4:
            raise ValueError(f"BigST expects input with shape [B, N, T, C], got {input.shape}")
        if input.shape[2] != self.seq_len:
            raise ValueError(f"BigST expects seq_len={self.seq_len}, got {input.shape[2]}")
        if input.shape[-1] != self.input_dim:
            raise ValueError(f"BigST expects input_dim={self.input_dim}, got {input.shape[-1]}")

        temporal = self._temporal_features(input, input_mark)
        return torch.cat([input, temporal], dim=-1)

    def _temporal_features(self, input, input_mark):
        batch_size, num_nodes, seq_len, _ = input.shape
        if input_mark is None:
            temporal = input.new_zeros(batch_size, num_nodes, seq_len, 2)
            if seq_len > 1:
                temporal[..., 0] = torch.arange(seq_len, device=input.device, dtype=input.dtype) / seq_len
            return temporal

        if input_mark.dim() == 3:
            mark = input_mark[:, -seq_len:, -2:].unsqueeze(1).expand(-1, num_nodes, -1, -1)
        elif input_mark.dim() == 4:
            mark = input_mark[:, :, -seq_len:, -2:]
        else:
            raise ValueError(f"BigST expects input_mark with 3 or 4 dims, got {input_mark.shape}")
        if mark.shape[-1] < 2:
            raise ValueError(f"BigST expects at least two temporal features, got {mark.shape}")
        return mark.to(device=input.device, dtype=input.dtype)

    def _time_index(self, time_feature):
        if time_feature.numel() == 0:
            return time_feature.long()
        if torch.max(time_feature).detach() <= 1:
            idx = torch.floor(time_feature * self.time_num)
        else:
            idx = torch.floor(time_feature)
        return idx.long().clamp(0, self.time_num - 1)

    def _week_index(self, week_feature):
        if week_feature.numel() == 0:
            return week_feature.long()
        if torch.max(week_feature).detach() <= 1:
            idx = torch.floor(week_feature * self.week_num)
        else:
            idx = torch.floor(week_feature)
        return idx.long().clamp(0, self.week_num - 1)


class LinearizedConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, tau, random_feature_dim):
        super().__init__()
        self.input_fc = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
        self.output_fc = nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1))
        self.activation = nn.Sigmoid()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.conv_app_layer = ConvApproximation(tau, random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.activation(self.input_fc(input_data)) * self.output_fc(input_data)
        x = self.dropout_layer(x).permute(0, 2, 3, 1)
        x = self.conv_app_layer(x, node_vec1, node_vec2)
        return x.permute(0, 3, 1, 2)


class ConvApproximation(nn.Module):
    def __init__(self, tau, random_feature_dim):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim

    def forward(self, x, node_vec1, node_vec2):
        dim = node_vec1.shape[-1]
        seed = int(torch.ceil(torch.abs(torch.sum(node_vec1.detach()) * 1e8)).item())
        random_matrix = create_random_matrix(self.random_feature_dim, dim, seed, node_vec1.device)
        node_vec1_prime = random_feature_map(node_vec1 / math.sqrt(self.tau), True, random_matrix)
        node_vec2_prime = random_feature_map(node_vec2 / math.sqrt(self.tau), False, random_matrix)
        return linear_kernel(x, node_vec1_prime, node_vec2_prime)


def create_random_matrix(m, d, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    blocks = []
    full_blocks = int(m / d)
    for _ in range(full_blocks):
        q, _ = torch.linalg.qr(torch.randn((d, d), generator=generator))
        blocks.append(q.t())
    remaining_rows = m - full_blocks * d
    if remaining_rows > 0:
        q, _ = torch.linalg.qr(torch.randn((d, d), generator=generator))
        blocks.append(q.t()[:remaining_rows])
    final_matrix = torch.vstack(blocks)
    multiplier = torch.norm(torch.randn((m, d), generator=generator), dim=1)
    return torch.matmul(torch.diag(multiplier), final_matrix).to(device)


def random_feature_map(data, is_query, projection_matrix, numerical_stabilizer=1e-6):
    data_normalizer = 1.0 / math.sqrt(math.sqrt(data.shape[-1]))
    data = data_normalizer * data
    ratio = 1.0 / math.sqrt(projection_matrix.shape[0])
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.sum(torch.square(data), dim=-1).unsqueeze(-1) / 2.0
    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True)[0])
            + numerical_stabilizer
        )
    else:
        max_data = torch.max(torch.max(data_dash, dim=-1, keepdim=True)[0], dim=-3, keepdim=True)[0]
        data_dash = ratio * (torch.exp(data_dash - diag_data - max_data) + numerical_stabilizer)
    return data_dash


def linear_kernel(x, node_vec1, node_vec2):
    node_vec1 = node_vec1.permute(1, 0, 2, 3)
    node_vec2 = node_vec2.permute(1, 0, 2, 3)
    x = x.permute(1, 0, 2, 3)

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)

    one_matrix = torch.ones([node_vec2.shape[0]], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum).unsqueeze(-1)

    return out1.permute(1, 0, 2, 3) / out2.permute(1, 0, 2, 3).clamp_min(1e-6)
