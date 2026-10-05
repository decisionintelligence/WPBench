import torch
import torch.nn as nn

from ts_benchmark.baselines.st_model.layers.PatchSTG_model import WindowAttBlock


class PatchSTG(nn.Module):
    def __init__(self, configs):
        super(PatchSTG, self).__init__()
        self.node_num = configs.series_num
        self.ori_parts_idx = configs.ori_parts_idx
        self.reo_parts_idx = configs.reo_parts_idx
        self.reo_all_idx = configs.reo_all_idx
        self.spa_patchnum = configs.spa_patchnum
        self.spa_patchsize = configs.spa_patchsize
        self.factors = configs.factors
        self.node_dims = configs.node_dims
        self.tod_dims = configs.tod_dims
        self.dow_dims = configs.dow_dims
        self.tem_patchnum = configs.tem_patchnum
        self.tem_patchsize = configs.tem_patchsize
        self.layers = configs.layers
        self.tod = configs.tod
        self.dow = configs.dow

        # model_dims = input_emb + spa_emb + tem_emb
        dims = configs.input_dims + configs.tod_dims + configs.dow_dims + configs.node_dims

        # spatio-temporal embedding -> section 4.1 in paper
        # input_emb
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=configs.input_dims, kernel_size=(1, configs.tem_patchsize), stride=(1, configs.tem_patchsize), bias=True)
        # spa_emb
        self.node_emb = nn.Parameter(
                torch.empty(self.node_num, self.node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        # tem_emb
        self.time_in_day_emb = nn.Parameter(
                torch.empty(self.tod, self.tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(
                torch.empty(self.dow, self.dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # dual attention encoder -> section 4.3 in paper, factors for merging the leaf nodes of KDTree
        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(dims, 1, self.spa_patchnum//self.factors, self.spa_patchsize* self.factors, mlp_ratio=1) for _ in range(self.layers)
        ])

        # projection decoder -> section 4.4 in paper
        self.regression_conv = nn.Conv2d(in_channels=self.tem_patchnum*dims, out_channels=self.tem_patchsize*self.tem_patchnum, kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # spatio-temporal embedding -> section 4.1 in paper
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points

        # dual attention encoder -> section 4.3 in paper
        for block in self.spa_encoder:
            rex = block(rex)

        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices

        # projection decoder -> section 4.4 in paper
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        b,t,n,_ = x.shape

        # input traffic + time of day + day of week as the input signal
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]

        # cat time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # cat day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # cat spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data