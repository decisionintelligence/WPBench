import numpy as np
import math
import torch
import torch.nn as nn

def get_frequency_modes(seq_len, modes=4, mode_select_method='random'):
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean*nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


# class FourierBlock(nn.Module):
#     def __init__(self, node, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
#         super(FourierBlock, self).__init__()
#         self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(node, in_channels, out_channels, len(self.index), dtype=torch.cfloat))
#
#     def compl_mul1d(self, input, weights):
#         return torch.einsum("bhi,hio->bho", input, weights)
#
#     def forward(self, q):
#         B, D, N, L = q.shape
#         x = q.permute(0, 2, 1, 3)
#         x_ft = torch.fft.rfft(x, dim=-1)
#         out_ft = torch.zeros(B, N, D, L // 2 + 1, device=x.device, dtype=torch.cfloat)
#         for wi, i in enumerate(self.index):
#             out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
#
#         output = torch.fft.irfft(out_ft, n=x.size(-1)).permute(0, 2, 1, 3)
#         return (output, None)
class FourierBlock(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        self.scale = (1 / (in_channels * out_channels))
        # 这里的 in_channels 和 out_channels 对应图卷积的 order
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    def forward(self, q):
        # q 的形状现为: [B, order, N, L, D]
        B, order, N, L, D = q.shape

        # 将时间维度 L 放到最后进行 rfft -> 形状: [B, N, D, order, L]
        x = q.permute(0, 2, 4, 1, 3)
        x_ft = torch.fft.rfft(x, dim=-1)

        # 准备输出频域张量 -> 形状: [B, N, D, order, L // 2 + 1]
        out_ft = torch.zeros(B, N, D, order, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        for wi, i in enumerate(self.index):
            # x_ft_slice 形状: [B, N, D, in_order]
            # weight 形状: [N, in_order, out_order]
            # 作用: 针对每一个被选中的频率 i，将特征从 in_order 映射到 out_order
            out_ft[:, :, :, :, wi] = torch.einsum("bndi,nio->bndo", x_ft[:, :, :, :, i], self.weights1[:, :, :, wi])

        # 逆傅里叶变换
        output = torch.fft.irfft(out_ft, n=x.size(-1))
        # 还原回原来的维度顺序 -> [B, order, N, L, D]
        output = output.permute(0, 3, 1, 4, 2)
        return (output, None)


class FourierCrossAttention(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len_q, seq_len_kv,
                 modes=64, mode_select_method='random', activation='tanh', policy=0):
        super(FourierCrossAttention, self).__init__()
        print('Fourier enhanced cross attention is used.')
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_q = get_frequency_modes(seq_len_q, modes=modes, mode_select_method=mode_select_method)
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)

        print('modes_q={}, index_q={}'.format(len(self.index_q), self.index_q))
        print('modes_kv={}, index_kv={}'.format(len(self.index_kv), self.index_kv))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask=None):
        B, E, H, L = q.shape
        xq = q.permute(0, 2, 1, 3) # size = [B, H, E, L]
        xk = k.permute(0, 2, 1, 3)
        xv = v.permute(0, 2, 1, 3)

        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        output = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 2, 1, 3)
        return (output, None)


class FourierSqeAttentionLayer(nn.Module):
    def __init__(self, node, in_channels, out_channels, seq_len_q, seq_len_kv, modes=64,
                 mode_select_method='random', activation='tanh', policy=0):
        super(FourierSqeAttentionLayer, self).__init__()
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.index_kv = get_frequency_modes(seq_len_kv, modes=modes, mode_select_method=mode_select_method)
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(node, in_channels, out_channels, len(self.index_q), dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        return torch.einsum("bhi,hio->bho", input, weights)

    def forward(self, q, k, v, mask=None):
        B, E, H, L = q.shape
        xq = q.permute(0, 2, 1, 3)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 1, 3)
        xv = v.permute(0, 2, 1, 3)
        xq_ft_ = torch.zeros(B, H, E, len(self.index_q), device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        for i, j in enumerate(self.index_q):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, len(self.index_kv), device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        for i, j in enumerate(self.index_kv):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]

        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))

        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)

        for i, j in enumerate(self.index_q):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        output = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1)).permute(0, 2, 1, 3)

        return (output, None)


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, Fourier_option, attention_option, modes, activation,
                 order=4, non_linear='linear', d_ff=None, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        # self.weight = nn.Parameter(torch.rand(1, order, 1, time_step, self.multi * self.time_step).cuda())
        self.weight = nn.Parameter(torch.rand(order, time_step, self.multi * self.time_step))
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        self.modes = modes
        self.non_linear = non_linear
        self.activation = activation
        self.attopt = attention_option
        if Fourier_option == 'FB':
            self.Fourier = FourierBlock(node=self.unit, in_channels=order, out_channels=order, seq_len=time_step, modes=self.modes, mode_select_method='random')
        # if Fourier_option == 'FCA':
        #     self.Fourier = FourierCrossAttention(node=self.unit, in_channels=order, out_channels=order,
        #                                          seq_len_q=time_step, seq_len_kv=2, modes=self.modes,
        #                                          mode_select_method='random')
        self.dropout = nn.Dropout(0.1)
        moving_avg = [2]
        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
        # t_embed = 4 * self.unit
        # out = 8
        # if self.attopt == 'fourier':
        #     self.feedforward = nn.Linear(t_embed, t_embed)
        #     self.linearq = nn.Linear(t_embed, out)
        #     self.lineark = nn.Linear(t_embed, out)
        #     self.linearv = nn.Linear(t_embed, out)
        #     self.attentionlayer = FourierSqeAttentionLayer(node=self.unit, in_channels=4, out_channels=4,
        #                                                    seq_len_q=time_step, seq_len_kv=time_step, modes=self.modes,
        #                                                    mode_select_method='random', activation=self.activation)
        #     self.lineart = nn.Linear(t_embed, t_embed)

    def compl_mul1d(self, input, weights):
        return torch.einsum("bni,nio->bno", input, weights)

    # def spe_seq_cell(self, input):
    #     batch_size, k, input_channel, node_cnt, time_step = input.size()
    #     input = input.view(batch_size, -1, node_cnt, time_step)
    #     new_x, _ = self.Fourier(input)
    #     x = input + self.dropout(new_x)
    #     xt = x.view(batch_size, time_step, -1)
    #     x_s, x_t = self.decomp1(xt)
    #
    #     # if self.attopt == 'fourier':
    #     #     x_s_o = x_s
    #     #     x_s = self.dropout(x_s.reshape(batch_size, -1, node_cnt, time_step))
    #     #     x_s_a, _ = self.attentionlayer(input, x_s, x_s)
    #     #     x_s_a = x_s_a.reshape(batch_size, time_step, -1)
    #     #     x_s = x_s_o + x_s_a
    #     #     x = x_s + x_t
    #
    #     if self.attopt == 'linear':
    #         x = x_s + x_t
    #     x = x.reshape(batch_size, -1, node_cnt, time_step)
    #     return x
    def spe_seq_cell(self, gfted):
        # gfted 形状: [B, order, N, L, D] (L = time_step * multi)
        B, order, N, L, D = gfted.shape

        new_x, _ = self.Fourier(gfted)
        x = gfted + self.dropout(new_x)

        # 时序分解：提取趋势和周期。decomp 需要时间 L 在倒数第二维，特征在最后一维(1)
        # 将张量重塑为 [Batch*order*N*D, L, 1] 批量送入 moving_avg
        xt = x.permute(0, 1, 2, 4, 3).reshape(-1, L, 1)
        x_s, x_t = self.decomp1(xt)

        # 恢复回 5D 张量 -> [B, order, N, L, D]
        x_s = x_s.reshape(B, order, N, D, L).permute(0, 1, 2, 4, 3)
        x_t = x_t.reshape(B, order, N, D, L).permute(0, 1, 2, 4, 3)

        if self.attopt == 'linear':
            x = x_s + x_t
        return x

    # def forward(self, x, mul_L):
    #     mul_L = mul_L.unsqueeze(1)
    #     x = x.unsqueeze(1)
    #     gfted = torch.matmul(mul_L, x)
    #     gfted = torch.matmul(gfted, self.weight)
    #     igfted = self.spe_seq_cell(gfted).unsqueeze(2)
    #     igfted = torch.sum(igfted, dim=1)
    #     if self.non_linear == 'linear':
    #         forecast_source = self.forecast(igfted).squeeze(1)
    #     elif self.non_linear == 'sigmoid':
    #         forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
    #     elif self.non_linear == 'relu':
    #         forecast_source = torch.relu(self.forecast(igfted).squeeze(1))
    #     elif self.non_linear == 'tanh':
    #         forecast_source = torch.tanh(self.forecast(igfted).squeeze(1))
    #     forecast = self.forecast_result(forecast_source)
    #     backcast_short = self.backcast_short_cut(x).squeeze(1)
    #     backcast_source = torch.sigmoid(backcast_short - self.backcast(igfted))
    #     x_back = torch.sigmoid(self.backcast(igfted))
    #     return forecast, backcast_source, x_back
    def forward(self, x, mul_L):
        # x 的输入形状: [B, N, T, D]
        # mul_L 的输入形状: [order, N, N]

        # 【修改3】图卷积：通过 einsum 在节点维度 (m, n) 上精准聚合
        # 输出形状: [B, order, N, T, D]
        gfted = torch.einsum('onm, bmtd -> bontd', mul_L, x)

        # 【修改4】时间维度线性投影：将 T 映射到 multi*T
        # 输出形状: [B, order, N, multi*T, D]
        gfted = torch.einsum('bontd, otm -> bonmd', gfted, self.weight)

        # 频域过滤和时序分解 -> 形状保持: [B, order, N, multi*T, D]
        igfted = self.spe_seq_cell(gfted)

        # 在 order (切比雪夫/Gegenbauer阶数) 维度聚合求和 -> 形状: [B, N, multi*T, D]
        igfted = torch.sum(igfted, dim=1)

        # 【修改5】处理全连接层 nn.Linear
        # nn.Linear 默认作用在张量的最后一个维度上，而我们要变换的是时间维度 (multi*T)
        # 所以必须把时间维度换到最后面 -> [B, N, D, multi*T]
        igfted_T = igfted.permute(0, 1, 3, 2)

        if self.non_linear == 'linear':
            forecast_source = self.forecast(igfted_T)
        elif self.non_linear == 'sigmoid':
            forecast_source = torch.sigmoid(self.forecast(igfted_T))
        elif self.non_linear == 'relu':
            forecast_source = torch.relu(self.forecast(igfted_T))
        elif self.non_linear == 'tanh':
            forecast_source = torch.tanh(self.forecast(igfted_T))

        forecast = self.forecast_result(forecast_source)  # 形状: [B, N, D, T]

        # 处理 backcast，同样要把 x 的时间维度移到最后
        x_T = x.permute(0, 1, 3, 2)  # [B, N, D, T]
        backcast_short = self.backcast_short_cut(x_T)  # 形状: [B, N, D, T]
        backcast_source = torch.sigmoid(backcast_short - self.backcast(igfted_T))
        x_back = torch.sigmoid(self.backcast(igfted_T))

        # 最后，把时间维度和特征维度换回来 -> [B, N, T, D]
        forecast = forecast.permute(0, 1, 3, 2)
        backcast_source = backcast_source.permute(0, 1, 3, 2)
        x_back = x_back.permute(0, 1, 3, 2)

        return forecast, backcast_source, x_back