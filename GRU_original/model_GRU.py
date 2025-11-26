# model_gru_light.py
import torch
import torch.nn as nn

class DepthwiseConv1d(nn.Module):
    """Depthwise conv block similar to TensorFlow DepthwiseConv1D"""
    def __init__(self, in_channels, kernel_size=7, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels,     # depthwise
            bias=bias
        )
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C) → conv expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.act(x)


class LightGRUDecoder(nn.Module):
    """
    Lightweight NeuralDecoder GRU (3 BiGRU layers + DepthwiseConv)
    Input dims must match your preprocessed features (1280)
    """

    def __init__(self,
                 input_dim=1280,
                 hidden_size=256,
                 num_layers=3,
                 num_classes=41):
        super().__init__()

        # 1) optional depthwise conv like original code
        self.conv = DepthwiseConv1d(in_channels=input_dim, kernel_size=7)

        # 2) 3-layer BiGRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # 3) FC → phoneme logits
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        """
        x: (B, T, C)
        lengths: list/ tensor of original time-lengths
        """

        # conv layer
        x = self.conv(x)

        # GRU requires packed padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_out, _ = self.gru(packed)

        # unpack
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )
        # out: (B, T, 2H)

        logits = self.fc(out)  # (B, T, num_classes)

        # CTC expects (T, B, C)
        return logits.permute(1, 0, 2)


class GRU6(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.fc(out)
        return logits.permute(1, 0, 2)   # (T,B,C)
