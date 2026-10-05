import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from ts_benchmark.baselines.st_model.layers.STSSDL_model import ADCRNN_Encoder, ADCRNN_Decoder


class STSSDL(nn.Module):
    def __init__(self, configs):
        super(STSSDL, self).__init__()
        self.num_nodes = configs.series_num
        self.input_dim = 1
        self.rnn_units = configs.rnn_units
        self.output_dim = configs.output_dim
        self.horizon = configs.pred_len
        self.rnn_layers = configs.num_layers
        self.cheb_k = configs. cheb_k
        self.ycov_dim = configs.ycov_dim
        self.tod_embed_dim = configs.tod_embed_dim
        self.cl_decay_steps = configs.cl_decay_steps
        self.use_curriculum_learning = configs.use_curriculum_learning
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_STE = configs.use_STE
        self.seq_len = configs.seq_len
        self.TDAY = int((24 * 60) / configs.time_slice_size)
        self.adaptive_embedding_dim = configs.adaptive_embedding_dim
        self.node_embedding_dim = configs.node_embedding_dim
        self.input_embedding_dim = configs.input_embedding_dim
        self.total_embedding_dim = self.tod_embed_dim + self.adaptive_embedding_dim + self.node_embedding_dim
        # prototypes
        self.prototype_num = configs.prototype_num
        self.prototype_dim = configs.prototype_dim
        self.prototypes = self.construct_prototypes()

        # projection & spatio-temporal embedding
        if self.use_STE:
            if self.adaptive_embedding_dim > 0:
                self.adaptive_embedding = nn.init.xavier_uniform_(
                    nn.Parameter(torch.empty(self.seq_len, self.num_nodes, self.adaptive_embedding_dim))
                )
            self.input_proj = nn.Linear(self.input_dim, self.input_embedding_dim)
            self.node_embedding = nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
            self.time_embedding = nn.Parameter(torch.empty(self.TDAY, self.tod_embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        # encoder
        self.adj_mx = configs.adj_mx
        if self.use_STE:
            self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_embedding_dim + self.total_embedding_dim,
                                          self.rnn_units, self.cheb_k, self.rnn_layers, len(self.adj_mx))
        else:
            self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers,
                                          len(self.adj_mx))

        # decoder
        self.decoder_dim = self.rnn_units + self.prototype_dim
        if self.use_STE:
            self.decoder = ADCRNN_Decoder(self.num_nodes,
                                          self.input_embedding_dim + self.total_embedding_dim - self.adaptive_embedding_dim,
                                          self.decoder_dim, self.cheb_k, self.rnn_layers, 1)
        else:
            self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim,
                                          self.cheb_k, self.rnn_layers, 1)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

        # graph
        self.hypernet = nn.Sequential(nn.Linear(self.decoder_dim * 2, self.tod_embed_dim, bias=True))

        self.act_dict = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(), 'sigmoid': nn.Sigmoid()}
        self.act_fn = 'sigmoid'  # 'relu' 'lrelu' 'sigmoid'

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_prototypes(self):
        prototypes_dict = nn.ParameterDict()
        prototype = torch.randn(self.prototype_num, self.prototype_dim)
        prototypes_dict['prototypes'] = nn.Parameter(prototype, requires_grad=True)  # (M, d)
        prototypes_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.prototype_dim),
                                             requires_grad=True)  # project to query
        for param in prototypes_dict.values():
            nn.init.xavier_normal_(param)

        return prototypes_dict

    def query_prototypes(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.prototypes['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.prototypes['prototypes'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.prototypes['prototypes'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.prototypes['prototypes'][ind[:, :, 0]]  # B, N, d
        neg = self.prototypes['prototypes'][ind[:, :, 1]]  # B, N, d
        mask = torch.stack([ind[:, :, 0], ind[:, :, 1]], dim=-1)  # B, N, 2

        return value, query, pos, neg, mask

    def calculate_distance(self, pos, pos_his, mask=None):
        score = torch.sum(torch.abs(pos - pos_his), dim=-1)
        return score, mask

    def forward(self, x, x_cov, x_his, y_cov, labels=None, batches_seen=None):
        if self.use_STE:
            if self.input_embedding_dim > 0:
                x = self.input_proj(x)  # [B,T,N,1]->[B,T,N,D]
            features = [x]

            if self.tod_embed_dim > 0:
                tod = (x_cov.squeeze(-1) * self.TDAY).long().clamp(0, self.TDAY - 1)
                time_emb = self.time_embedding[tod]  # [B, T, N, d]
                features.append(time_emb)
            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(
                    size=(x.shape[0], *self.adaptive_embedding.shape)
                )
                features.append(adp_emb)
            if self.node_embedding_dim > 0:
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(x.shape[0], x.shape[1], -1,
                                                                                -1)  # [B,T,N,d]
                features.append(node_emb)
            x = torch.cat(features, dim=-1)  # [B, T, N, D+d+80]
        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)
        v_t, q_t, p_t, n_t, mask = self.query_prototypes(h_t)
        if self.use_STE:
            if self.input_embedding_dim > 0:
                x_his = self.input_proj(x_his)  # [B,T,N,1]->[B,T,N,D]
            features = [x_his]
            if self.tod_embed_dim > 0:
                tod = (x_cov.squeeze(-1) * self.TDAY).long().clamp(0, self.TDAY - 1)
                time_emb = self.time_embedding[tod]  # [B, T, N, d]

                features.append(time_emb)
            if self.adaptive_embedding_dim > 0:
                adp_emb = self.adaptive_embedding.expand(
                    size=(x.shape[0], *self.adaptive_embedding.shape)
                )
                features.append(adp_emb)
            if self.node_embedding_dim > 0:
                node_emb = self.node_embedding.unsqueeze(0).unsqueeze(1).expand(x.shape[0], x.shape[1], -1,
                                                                                -1)  # [B,T,N,d]
                features.append(node_emb)
            x_his = torch.cat(features, dim=-1)  # [B, T, N, D+d+80]
        h_his_en, state_his_en = self.encoder(x_his, init_state, supports_en)  # B, T, N, hidden
        h_a = h_his_en[:, -1, :, :]  # B, N, hidden (last state)
        v_a, q_a, p_a, n_a, mask_his = self.query_prototypes(h_a)

        latent_dis, _ = self.calculate_distance(q_t, q_a)
        prototype_dis, mask_dis = self.calculate_distance(p_t, p_a)

        query = torch.stack([q_t, q_a], dim=0)
        pos = torch.stack([p_t, p_a], dim=0)
        neg = torch.stack([n_t, n_a], dim=0)
        mask = torch.stack([mask, mask_his], dim=0) if mask is not None else [None, None]

        h_de = torch.cat([h_t, v_t], dim=-1)
        h_aug = torch.cat([h_t, v_t, h_a, v_a], dim=-1)  # B, N, D

        node_embeddings = self.hypernet(h_aug)  # B, N, e
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1)
        supports_de = [support]

        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)

        out = []
        for t in range(self.horizon):
            if self.use_STE:
                if self.input_embedding_dim > 0:
                    go = self.input_proj(go)  # equal to torch.zeros(B,N,D)
                features = [go]
                if self.tod_embed_dim > 0:
                    tod = (y_cov[:, t, ...].squeeze(-1) * self.TDAY).long().clamp(0, self.TDAY - 1)
                    time_emb = self.time_embedding[tod]
                    features.append(time_emb)
                if self.node_embedding_dim > 0:
                    node_emb = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B,N,d]
                    features.append(node_emb)
                go = torch.cat(features, dim=-1)  # [B, T, N, D+d]
                h_de, ht_list = self.decoder(go, ht_list, supports_de)
            else:
                h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports_de)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)

        return output, query, pos, neg, mask, latent_dis, prototype_dis


