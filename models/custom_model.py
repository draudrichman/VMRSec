# models/custom_model.py
import torch
import torch.nn as nn
from nncore.nn import MODELS

@MODELS.register()
class CustomModel(nn.Module):
    def __init__(self, video_dim, audio_dim, hidden_dim, num_classes):
        """
        Args:
            video_dim (int): Dimension of video features.
            audio_dim (int): Dimension of audio features.
            hidden_dim (int): Dimension of hidden layers.
            num_classes (int): Number of action classes.
        """
        super(CustomModel, self).__init__()
        
        # Video encoder
        self.video_encoder = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Audio encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, video, audio):
        # Encode video and audio
        video_encoded = self.video_encoder(video)
        audio_encoded = self.audio_encoder(audio)
        
        # Fuse modalities
        fused = torch.cat((video_encoded, audio_encoded), dim=-1)
        fused = self.fusion(fused)
        
        # Predict actions
        logits = self.classifier(fused)
        return logits