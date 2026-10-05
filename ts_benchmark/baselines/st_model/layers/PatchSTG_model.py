import torch.nn as nn

from timm.models.vision_transformer import Attention, Mlp


class WindowAttBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        self.nnorm1 = nn.LayerNorm(hidden_size)
        self.nattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size)
        self.sattn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        B, T, _, D = x.shape
        # P: ptach num and N: patch size
        P, N = self.num, self.size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # depth attention
        qkv = self.snorm1(x.reshape(B * T * P, N, D))
        x = x + self.sattn(qkv).reshape(B, T, P, N, D)
        x = x + self.smlp(self.snorm2(x))

        # breadth attention
        qkv = self.nnorm1(x.transpose(2, 3).reshape(B * T * N, P, D))
        x = x + self.nattn(qkv).reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))

        return x.reshape(B, T, -1, D)