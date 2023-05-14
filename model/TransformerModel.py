import torch
import torch.nn as nn
import math


class TransformerEmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=8):
        super(TransformerEmbeddingModel, self).__init__()

        self.embed_dim = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 将输入张量转换为 (seq_len, batch_size, input_dim) 形状
        x = x.unsqueeze(0)

        # 将输入张量进行嵌入
        x = self.embedding(x)

        # 将嵌入后的张量传递给 Transformer 编码器
        x = self.encoder(x)

        # 取得序列编码的平均值
        x = x.mean(dim=0)

        # 传递给输出层进行分类
        out = self.out(x)
        # print(out.shape)

        return out

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

