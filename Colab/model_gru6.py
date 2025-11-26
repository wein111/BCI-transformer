# model_gru_light.py
import torch
import torch.nn as nn


class DepthwiseConv1d(nn.Module):
    """
    Depthwise conv block (类似 TF 的 DepthwiseConv1D)
    输入:  (B, T, C)
    输出:  (B, T, C)
    """
    def __init__(self, in_channels, kernel_size=7, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,  # depthwise
            bias=bias,
        )
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C) → conv 要 (B, C, T)
        x = x.transpose(1, 2)     # (B, C, T)
        x = self.conv(x)          # (B, C, T)
        x = x.transpose(1, 2)     # (B, T, C)
        return self.act(x)


class LightGRUDecoder(nn.Module):
    """
    强化版 NeuralDecoder-ish GRU 后端:
      - 前置 DepthwiseConv1d
      - 6 层 BiGRU
      - LayerNorm + dropout
      - CTC 输出 logits

    输入:
      x:        (B, T, C)   这里 C 要对应你预处理后的 1280
      lengths:  每个样本的有效时间步长 (T_new)

    输出:
      logits:   (T, B, num_classes)  (CTC 要求格式)
    """

    def __init__(
        self,
        input_dim=1280,
        hidden_size=512,
        num_layers=6,
        num_classes=41,
        conv_kernel=7,
        dropout=0.2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 1) 简单的 LayerNorm 在通道维上，稳定输入分布
        self.input_ln = nn.LayerNorm(input_dim)

        # 2) depthwise conv（跟原版类似的时域滤波）
        self.conv = DepthwiseConv1d(in_channels=input_dim, kernel_size=conv_kernel)

        # 3) 多层双向 GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        # 4) 对 GRU 输出再做一次 LayerNorm
        self.gru_ln = nn.LayerNorm(hidden_size * 2)

        # 5) 全连接映射到 phoneme 类别（含 CTC blank）
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        # 模型初始化：仿照常用做法稍微 careful 一点
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0.0)

        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            self.fc.bias.data.fill_(0.0)

    def forward(self, x, lengths):
        """
        x:       (B, T, C)
        lengths: 1D Tensor, 每个样本的有效长度 (和 CTC 的 input_lengths 对应)
        """

        # LayerNorm + conv
        x = self.input_ln(x)   # (B, T, C)
        x = self.conv(x)       # (B, T, C)

        # pack 序列，按 lengths 截断
        # 注意 lengths 必须在 CPU 上
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths_cpu, batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        # pad 回来
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, T_max, 2H)

        out = self.gru_ln(out)

        logits = self.fc(out)  # (B, T_max, num_classes)

        # CTC 要 (T, B, C)
        return logits.permute(1, 0, 2)
