import torch
import torch.nn as nn

from ts_benchmark.baselines.st_model.layers.STWave_model import Dual_Enconder, Adaptive_Fusion, FeedForward, TemEmbedding



class STWave(nn.Module):
    def __init__(self, configs):
        super(STWave, self).__init__()
        features = configs.heads * configs.dims
        I = torch.arange(configs.localadj.shape[0]).unsqueeze(-1)
        localadj = torch.cat([I, torch.from_numpy(configs.localadj)], -1)
        self.input_len = configs.seq_len
        self.dual_enc = nn.ModuleList(
            [Dual_Enconder(configs.heads, configs.dims, configs.samples, configs.localadj, configs.spawave, configs.temwave) for i in range(configs.num_layers)])
        self.adp_f = Adaptive_Fusion(configs.heads, configs.dims)

        self.pre_l = nn.Conv2d(configs.seq_len, configs.pred_len, (1, 1))
        self.pre_h = nn.Conv2d(configs.seq_len, configs.pred_len, (1, 1))
        self.pre = nn.Conv2d(configs.seq_len, configs.pred_len, (1, 1))

        self.start_emb_l = FeedForward([configs.enc_in, features, features])
        self.start_emb_h = FeedForward([configs.enc_in, features, features])
        self.end_emb = FeedForward([features, features, configs.series_dim])
        self.end_emb_l = FeedForward([features, features, configs.series_dim])
        self.te_emb = TemEmbedding(features)

    def forward(self, XL, XH, TE):
        '''
        XL: [B,T,N,F]
        XH: [B,T,N,F]
        TE: [B,T,2]
        return: [B,T,N,1]
        '''
        xl, xh = self.start_emb_l(XL), self.start_emb_h(XH)
        te = self.te_emb(TE)

        for enc in self.dual_enc:
            xl, xh = enc(xl, xh, te[:, :self.input_len, :, :])

        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)
        hat_y = self.adp_f(hat_y_l, hat_y_h, te[:, self.input_len:, :, :])
        hat_y, hat_y_l = self.end_emb(hat_y), self.end_emb_l(hat_y_l)

        return hat_y, hat_y_l
