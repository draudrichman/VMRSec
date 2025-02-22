# models/custom_model.py
import torch
import torch.nn as nn
from nncore.nn import MODELS


@MODELS.register()
class CustomModel(nn.Module):
    def __init__(self, video_dim, hidden_dim, num_classes):
        super(CustomModel, self).__init__()
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, video):
        video_encoded = self.video_encoder(video)
        logits = self.classifier(video_encoded)
        return logits