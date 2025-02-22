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


class TemporalActionModel(nn.Module):
    def __init__(self, video_dim, hidden_dim, num_classes, max_proposals=10):
        super(TemporalActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_proposals = max_proposals  # Max number of action segments per video

        # Temporal encoder (LSTM)
        self.lstm = nn.LSTM(video_dim, hidden_dim, batch_first=True)
        # Proposal generator
        self.proposal_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * max_proposals)  # 3 outputs per proposal: class, start, end
        )
        self.classifier = nn.Linear(hidden_dim, num_classes * max_proposals)

    def forward(self, video):
        # video: (batch_size, max_length, video_dim)
        lstm_out, _ = self.lstm(video)  # Shape: (batch_size, max_length, hidden_dim)
        # Average over time for simplicity (could use attention instead)
        lstm_avg = lstm_out.mean(dim=1)  # Shape: (batch_size, hidden_dim)

        # Predict proposals: (class_idx, start, end) for each proposal
        proposals = self.proposal_head(lstm_avg)  # Shape: (batch_size, 3 * max_proposals)
        proposals = proposals.view(-1, self.max_proposals, 3)  # Shape: (batch_size, max_proposals, 3)

        # Predict class scores for each proposal
        class_scores = self.classifier(lstm_avg)  # Shape: (batch_size, num_classes * max_proposals)
        class_scores = class_scores.view(-1, self.max_proposals, self.num_classes)  # Shape: (batch_size, max_proposals, num_classes)

        # Softmax over classes for each proposal
        class_probs = torch.softmax(class_scores, dim=-1)

        return proposals, class_probs  # (batch_size, max_proposals, 3), (batch_size, max_proposals, num_classes)