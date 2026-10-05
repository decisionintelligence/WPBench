import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.st_model.layers.TGGC_model import StockBlockLayer


class TGGC(nn.Module):
    def __init__(self, configs):
        super(TGGC, self).__init__()
        self.unit = configs.num_nodes
        self.non_linear = configs.non_linear
        self.stack_cnt = configs.stack_cnt
        self.alpha = configs.leaky_rate
        self.time_step = configs.seq_len
        self.horizon = configs.horizon
        self.Fouropt = configs.Fouropt
        self.attention_option = configs.attention_set
        self.modes = configs.modes
        self.order = configs.order
        self.activation = configs.activation
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step * configs.enc_in, self.unit)
        self.multi_layer = configs.num_layers
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(
                    self.time_step,
                    self.unit,
                    self.multi_layer,
                    order=self.order,
                    non_linear=self.non_linear,
                    Fourier_option=self.Fouropt,
                    attention_option=self.attention_option,
                    modes=self.modes,
                    activation=self.activation,
                    stack_cnt=i,
                )
                for i in range(self.stack_cnt)
            ]
        )
        self.fc_1 = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), int(self.time_step)),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=configs.dropout)
        self.gconv = configs.gconv
        self.coe_a = configs.coe_a
        self.coe_b = configs.coe_b
        self.feature_dim = 1
        self.to(configs.device)

    def get_laplacian(self, graph, normalize):

        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(
                graph.size(0), device=graph.device, dtype=graph.dtype
            ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    # def Mono_polynomial(self, laplacian):
    #     N = laplacian.size(0)
    #     laplacian = laplacian.unsqueeze(0)
    #     first_laplacian = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
    #     second_laplacian = first_laplacian - laplacian
    #     third_laplacian = torch.matmul(second_laplacian, second_laplacian)
    #     multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)
    #     return multi_order_laplacian
    #
    # def Bern_polynomial(self, laplacian):
    #     N = laplacian.size(0)
    #     laplacian = laplacian.unsqueeze(0)/2
    #     one_lapla = torch.ones([1, N, N], device=laplacian.device, dtype=torch.float)
    #     laplacian_1 = one_lapla - laplacian
    #     first_laplacian = torch.matmul(laplacian_1, laplacian_1)
    #     second_laplacian = 2 * torch.matmul(laplacian, laplacian_1)
    #     third_laplacian = 2 * torch.matmul(laplacian, laplacian)
    #     multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian], dim=0)
    #     return multi_order_laplacian

    def Cheb_polynomial(self, laplacian):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (
            2 * torch.matmul(laplacian, second_laplacian) - first_laplacian
        )
        forth_laplacian = (
            2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def Jacobi_coe(self, k, a=1, b=1, l=-1.0, r=1.0):
        theta_0 = (2 * k + a + b) * (2 * k + a + b - 1) / (2 * k * (k + a + b))
        theta_1 = (
            (2 * k + a + b - 1)
            * (a * a - b * b)
            / (2 * k * (k + a + b) * (2 * k + a + b - 2))
        )
        theta_2 = (
            (k + a - 1)
            * (k + b - 1)
            * (2 * k + a + b)
            / (k * (k + a + b) * (2 * k + a + b - 2))
        )
        return [theta_0, theta_1, theta_2]

    def Jacobi_polynomial(self, laplacian, a=1, b=1, l=-1.0, r=1.0):

        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = (a - b) / 2 + (a + b + 2) / 2 * laplacian
        Theta_2 = self.Jacobi_coe(2, a, b)
        third_laplacian = (
            torch.matmul((Theta_2[0] * laplacian + Theta_2[1]), second_laplacian)
            + Theta_2[2] * first_laplacian
        )
        Theta_3 = self.Jacobi_coe(3, a, b)
        forth_laplacian = (
            torch.matmul((Theta_3[0] * laplacian + Theta_3[1]), third_laplacian)
            + Theta_3[2] * second_laplacian
        )
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def Gegen_coe(self, k, a=1, l=-1.0, r=1.0):
        theta_0 = 2 * (k + a - 1)
        theta_1 = k + 2 * (a - 1)
        return [theta_0, theta_1]

    def Gegen_polynomial(self, laplacian, a=1, l=-1.0, r=1):
        N = laplacian.size(0)
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.ones(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        k = r
        if k == 1:
            multi_order_laplacian = first_laplacian
        if k == 2:
            second_laplacian = 2 * a * laplacian
            multi_order_laplacian = torch.cat(
                [first_laplacian, second_laplacian], dim=0
            )
        if k == 3:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = (
                1
                / 2
                * (
                    torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                    - Theta_2[1] * first_laplacian
                )
            )
            multi_order_laplacian = torch.cat(
                [first_laplacian, second_laplacian, third_laplacian], dim=0
            )
        if k == 4:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = (
                1
                / 2
                * (
                    torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                    - Theta_2[1] * first_laplacian
                )
            )
            Theta_3 = self.Gegen_coe(3, a)
            forth_laplacian = (
                1
                / 3
                * (
                    torch.matmul((Theta_3[0] * laplacian), third_laplacian)
                    - Theta_3[1] * second_laplacian
                )
            )
            multi_order_laplacian = torch.cat(
                [first_laplacian, second_laplacian, third_laplacian, forth_laplacian],
                dim=0,
            )
        if k == 5:
            second_laplacian = 2 * a * laplacian
            Theta_2 = self.Gegen_coe(2, a)
            third_laplacian = (
                1
                / 2
                * (
                    torch.matmul((Theta_2[0] * laplacian), second_laplacian)
                    - Theta_2[1] * first_laplacian
                )
            )
            Theta_3 = self.Gegen_coe(3, a)
            forth_laplacian = (
                1
                / 3
                * (
                    torch.matmul((Theta_3[0] * laplacian), third_laplacian)
                    - Theta_3[1] * second_laplacian
                )
            )
            Theta_4 = self.Gegen_coe(4, a)
            fifth_laplacian = (
                1
                / 3
                * (
                    torch.matmul((Theta_4[0] * laplacian), forth_laplacian)
                    - Theta_3[1] * third_laplacian
                )
            )
            multi_order_laplacian = torch.cat(
                [
                    first_laplacian,
                    second_laplacian,
                    third_laplacian,
                    forth_laplacian,
                    fifth_laplacian,
                ],
                dim=0,
            )
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        B, N, T, D = x.shape
        x_flat = x.reshape(B, N, T * D)

        gru_input = x_flat.permute(1, 0, 2).contiguous()

        input, _ = self.GRU(gru_input)

        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )

        if self.gconv == "cheby":
            mul_L = self.Cheb_polynomial(laplacian)
        elif self.gconv == "jacobi":
            mul_L = self.Jacobi_polynomial(laplacian, self.coe_a, self.coe_b)
        elif self.gconv == "gegen":
            mul_L = self.Gegen_polynomial(laplacian, self.coe_a, r=self.order)
        # elif self.gconv == 'bern':
        #     mul_L = self.Bern_polynomial(laplacian)
        # elif self.gconv == 'mono':
        #     mul_L = self.Mono_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, input):
        # if len(input.shape) == 3:
        #     # 假设 self.feature_dim 就是你的 D (在 __init__ 传入)
        #     input = input.unsqueeze(-1).repeat(1, 1, 1, getattr(self, 'feature_dim', 1))
        # # input形状标记：
        #
        mul_L, attention = self.latent_correlation_layer(input)
        # X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        X = input
        result = []
        back = []
        for stack_i in range(self.stack_cnt):
            forecast, X, x_back = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
            back.append(x_back)
        forecast = torch.stack(result).sum(0)  # B, N,T, D
        foreback = torch.stack(back).sum(0)

        forecast = forecast.permute(0, 1, 3, 2)
        forecast = self.fc_1(forecast)
        forecast = forecast.permute(0, 1, 3, 2)  # 最终变为目标形状: [B, N, H, D]

        foreback = foreback.permute(0, 1, 3, 2)
        foreback = self.fc_2(foreback)
        foreback = foreback.permute(0, 1, 3, 2)
        # if foreback.shape[-1] == 1:
        #     foreback = foreback.squeeze(-1)
        # if forecast.shape[-1] == 1:
        #     forecast = forecast.squeeze(-1)
        return forecast, attention, foreback
