"""
AudioNet v2 Model Architecture for Zero Rabies-MMNet.
2D Spectrogram Convolutional Neural Network with spatial attention and linear projection head.
Matches frozen audio model checkpoint (6,179,714 parameters).
"""

import torch
import torch.nn as nn


class AudioNet(nn.Module):
    """
    AudioNet v2 architecture committed by the audio branch developer.
    Operates on 64-mel log-power spectrograms of shape (B, 1, 64, 188).
    Outputs a single uncalibrated logit and a 128-dimensional embedding.
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.flatten_size = 64 * 16 * 47

        self.embed = nn.Sequential(
            nn.Linear(self.flatten_size, 128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        x: tensor of shape (batch, 1, 64, 188)
        Returns:
            logit: tensor of shape (batch,)
            emb: tensor of shape (batch, 128)
        """
        x = self.features(x)
        x = x * self.attention(x)
        x = x.view(x.size(0), -1)
        emb = self.embed(x)
        logit = self.head(self.dropout(emb)).squeeze(1)
        return logit, emb
