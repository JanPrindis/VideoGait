import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gaitDetectNN.utils.registry import MODELS

@MODELS.register
class GaitBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim=64, dense_units=32, num_layers=2, num_classes=4, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        lstm_out_size = hidden_dim * 2

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, dense_units),
            nn.ReLU(),
            nn.Linear(dense_units, num_classes)
        )

    def forward(self, x, lengths):
        # Pack sequence - ignore padding
        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM Pass
        lstm_out_packed, _ = self.lstm(x_packed)

        # Unpack
        # lstm_out: (Batch, Time, Hidden * 2)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Classification
        logits = self.head(lstm_out)

        return logits
