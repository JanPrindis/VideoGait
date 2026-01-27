import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gaitDetectNN.utils.registry import MODELS

@MODELS.register
class GaitBiGRU(nn.Module):
    def __init__(self, input_size, hidden_dim=128, dense_units=64, num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()

        # GRU Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # Classification head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x, lengths):
        # x shape: (Batch, Time, Features)
        # Packing
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # GRU Forward
        packed_out, _ = self.gru(packed_x)

        # Unpacking
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Classification Head
        logits = self.fc_head(out)

        return logits
