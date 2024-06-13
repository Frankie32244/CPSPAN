import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):    # input_dim 输入数据的特征维度,feature_dim 是输出数据特征的维度
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),             # 一个线性层，将输入维度从 input_dim 映射到 500
            nn.ReLU(),                             # 一个非线性激活函数，添加非线性特性
            nn.Linear(500, 500),                   # 另一个线性层，保持维度为 500
            nn.ReLU(),
            nn.Linear(500, 2000),  # 改            # 线性层，将维度从 500 映射到 2000
            nn.ReLU(),
            nn.Linear(2000, feature_dim),  # 改    # 最后一个线性层，将维度从 2000 映射到目标特征维度 
        )

    def forward(self, x):
        return self.encoder(x)                     # 将输入数据通过编码器网络进行处理，输出编码后的特征表示 


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),  # 改
            nn.ReLU(),
            nn.Linear(2000, 500),  # 改
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, views, input_size, feature_dim):   # views 视图的数量、input_size 每个视图的输入维度、feature_dim 编码后的特征维度
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        self.views = views                         
        for v in range(views):
            self.encoders.append(Encoder(input_size[v], feature_dim))
            self.decoders.append(Decoder(input_size[v], feature_dim))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, xs):
        zs = []
        xrs = []
        for v in range(self.views):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            zs.append(z)
            xrs.append(xr)
        return zs, xrs        # x 为原始数据、z(zs)为encoder之后的数据、y(xrs)为decoder之后的数据
