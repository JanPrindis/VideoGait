import torch
import torch.nn as nn
from gaitDetectNN.utils.registry import MODELS

@MODELS.register
class GaitLSTM(nn.Module):
    """
    An LSTM model for gait event detection, inspired by Zhang et al.
    Detecting Heel Strike and toe off Events Using Kinematic Methods and LSTM Models
    https://arxiv.org/pdf/2503.00794

    This model processes sequences of keypoint data to predict the independent
    probability of four different gait events (L_HS, L_TO, R_HS, R_TO) for each time step.
    """
    def __init__(self,
                 input_size: int,
                 num_classes: int = 4,
                 lstm_units: int = 128,
                 dense_units: int = 32,
                 dropout_prob: float = 0.3,
                 **kwargs):
        """
        Args:
            input_size (int): The number of features for each frame.
            num_classes (int): The number of output classes (default: 4 for the events).
            lstm_units (int): The number of units in the LSTM layer (from the paper).
            dense_units (int): The number of neurons in the intermediate Dense layer (from the paper).
            dropout_prob (float): The dropout probability for regularization (from the paper).
        """
        super().__init__()
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.intermediate_dense = nn.Linear(in_features=lstm_units, out_features=dense_units)
        self.relu = nn.ReLU()
        self.output_dense = nn.Linear(in_features=dense_units, out_features=num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # The paper's "Masking layer"
        # This tells the LSTM to ignore padded time steps
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, _ = self.lstm(packed_input)

        # Unpack the sequence to apply the dense layers to each time step individually
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.dropout(output)
        output = self.intermediate_dense(output)
        output = self.relu(output)
        logits = self.output_dense(output)

        return logits
