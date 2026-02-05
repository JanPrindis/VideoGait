import torch
import torch.nn as nn
import math
from utils.registry import MODELS

import warnings
warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage*"
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # (max_len, d_model) -> (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        # Add PE to x (slice to the current sequence length)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@MODELS.register
class GaitTransformer(nn.Module):
    """
    SOTA-style Transformer for Gait Event Detection.
    Combines Conv1D embedding (for local features) with Transformer Encoder (for global context).

    Architecture:
    Input -> Conv1D Projection -> Positional Encoding -> Transformer Encoder Layers -> Linear Head
    """

    def __init__(
            self,
            input_size,
            d_model=128,
            kernel_size=3,
            padding=1,
            n_head=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.1,
            num_classes=4,
            max_len=5000,
            **kwargs):
        super().__init__()

        self.d_model = d_model

        # Feature Projection (Conv1D acts as a "Tokenizer" for motion)
        # Input: (Batch, Channels, Time) -> Output: (Batch, d_model, Time)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_size, d_model, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Transformer Encoder
        # batch_first=True is crucial for modern PyTorch (B, T, C)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # GELU is often better than ReLU for Transformers (BERT/GPT style)
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization generally works well
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _create_padding_mask(lengths, max_len):
        """
        Creates a boolean mask where True indicates padding (to be ignored).
        Shape: (Batch, Seq_Len)
        """
        # Create a range [0, 1, ..., max_len-1]
        seq_range = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # (1, max_len)

        # Expand lengths to (Batch, 1)
        len_col = lengths.unsqueeze(1)  # (Batch, 1)

        # Mask is True where index >= length
        mask = seq_range >= len_col
        return mask

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Time)

        # Permute for Transformer (Batch, Time, d_model)
        x = x.permute(0, 2, 1)

        # Projection (Conv1D expects B, C, T)
        x = self.input_projection(x)  # -> (Batch, d_model, Time)

        # Permute back
        x = x.permute(0, 2, 1)

        # Padding mask
        mask = self._create_padding_mask(lengths, x.size(1))
        mask = mask.to(x.device)

        # Add Positional Encoding
        x = self.pos_encoder(x)

        # Transformer Encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)  # -> (Batch, Time, d_model)

        # Classification
        output = self.classifier(x)  # -> (Batch, Time, Num_Classes)

        # Zero out padding
        output[mask] = 0

        return output
