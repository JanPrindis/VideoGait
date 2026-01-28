import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from gaitDetectNN.utils.registry import MODELS


class TemporalBlock(nn.Module):
    """
    Non-Causal TCN block: Dilated Conv -> ReLU -> Dropout -> Dilated Conv -> ReLU -> Dropout
    Plus Residual connection
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Calculate padding based on dilation and kernel size
        padding = (kernel_size - 1) * dilation // 2

        # First convolution
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)

        # Downsample pro Residual connection, if we are changing the number of channels
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

@MODELS.register
class GaitTCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for gait event detection.

    This architecture uses a series of dilated 1D convolutions to capture long-range
    temporal dependencies in the input sequence. It is non-causal (looks at both past
    and future frames) which is suitable for offline analysis.

    Args:
        input_size (int): Number of input features per frame.
        num_channels (list[int], optional): List of channel sizes for each TCN layer.
                                            Defaults to [64, 64, 64].
        kernel_size (int, optional): Size of the convolution kernel. Must be an odd number
                                     to ensure symmetric padding preserves sequence length.
                                     Defaults to 7.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
        num_classes (int, optional): Number of output classes (events). Defaults to 4.
    """
    def __init__(self, input_size, num_channels=None, kernel_size=7, dropout=0.2, num_classes=4):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 64, 64]

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponential growth of "field of view"
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        # Classification head
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, lengths=None):
        # x shape: (Batch, Time, Features)
        # TCN (Batch, Features, Time)
        x = x.transpose(1, 2)

        y = self.network(x)

        # Back to (Batch, Time, Channels) for classification
        y = y.transpose(1, 2)

        logits = self.linear(y)

        # Mask out padding
        if lengths is not None:
            device = logits.device

            # Mask (Batch, Time, 1)
            mask = torch.arange(logits.size(1), device=device)[None, :] < lengths.to(device)[:, None]

            # Expand mask for classes (Batch, Time, Num_Classes)
            mask = mask.unsqueeze(-1).expand_as(logits)

            # Zero out values outside the mask
            logits = logits * mask.float()

        return logits
