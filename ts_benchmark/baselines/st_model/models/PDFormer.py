import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from logging import getLogger
from ts_benchmark.baselines.st_model.layers.PDFormer_model import TokenEmbedding, DataEmbedding, STEncoderBlock

from numexpr import set_vml_num_threads


class PDFormer(nn.Module):
    def __init__(self, configs):
        super().__init__(configs)
        self.num_nodes = configs.series_num
        self.feature_dim = configs.enc_in
        self.ext_dim = configs.ext_dim
        self.dtw_matrix = configs.dtw_matrix
        self.adj_mx = configs.adj_mx
        # sd_mx = data_feature.get('sd_mx')
        # sh_mx = data_feature.get('sh_mx')
        sd_mx = configs.sd_mx
        sh_mx = configs.sh_mx
        self._logger = getLogger()

        self.embed_dim = configs.embed_dim
        self.skip_dim = configs.skip_dim
        lape_dim = configs.lape_dim
        geo_num_heads = configs.geo_num_heads
        sem_num_heads = configs.sem_num_heads
        t_num_heads = configs.t_num_heads
        mlp_ratio = configs.mlp_ratio
        qkv_bias = configs.qkv_bias
        drop = configs.dropout
        attn_drop = configs.attn_dropout
        drop_path = configs.drop_path
        self.s_attn_size = configs.s_attn_size
        self.t_attn_size = configs.t_attn_size
        enc_depth = configs.enc_depth
        type_ln = configs.type_ln
        self.type_short_path = configs.type_short_path

        self.output_dim = configs.output_dim
        self.input_window = configs.seq_len
        self.output_window = configs.pred_len
        add_time_in_day = configs.add_time_in_day
        add_day_in_week = configs.add_day_in_week
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.far_mask_delta = configs.far_mask_delta
        self.dtw_delta = configs.dtw_delta


        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask[i]] = 0
            self.sem_mask = self.sem_mask.bool()

        self.pattern_keys = torch.from_numpy(configs.pattern_keys).float().to(self.device)
        self.pattern_embeddings = nn.ModuleList([
            TokenEmbedding(self.s_attn_size, self.embed_dim) for _ in range(self.output_dim)
        ])

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size, geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, t_num_heads=t_num_heads,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, type_ln=type_ln, output_dim=self.output_dim,
            ) for i in range(enc_depth)
        ])

        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch, lap_mx=None):
        x = batch['X']
        T =  x.shape[1]
        x_pattern_list = []
        for i in range(self.s_attn_size):
            x_pattern = F.pad(
                x[:, :T + i + 1 - self.s_attn_size, :, :self.output_dim],
                (0, 0, 0, 0, self.s_attn_size - 1 - i, 0),
                "constant", 0,
            ).unsqueeze(-2)
            x_pattern_list.append(x_pattern)
        x_patterns = torch.cat(x_pattern_list, dim=-2)  # (B, T, N, s_attn_size, output_dim)

        x_pattern_list = []
        pattern_key_list = []
        for i in range(self.output_dim):
            x_pattern_list.append(self.pattern_embeddings[i](x_patterns[..., i]).unsqueeze(-1))
            pattern_key_list.append(self.pattern_embeddings[i](self.pattern_keys[..., i]).unsqueeze(-1))
        x_patterns = torch.cat(x_pattern_list, dim=-1)
        pattern_keys = torch.cat(pattern_key_list, dim=-1)

        enc = self.enc_embed_layer(x, lap_mx)
        skip = 0
        for i, encoder_block in enumerate(self.encoder_blocks):
            enc = encoder_block(enc, x_patterns, pattern_keys, self.geo_mask, self.sem_mask)
            skip += self.skip_convs[i](enc.permute(0, 3, 2, 1))

        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1)

