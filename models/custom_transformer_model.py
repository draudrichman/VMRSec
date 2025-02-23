import torch
import torch.nn as nn

class TransformerActionModel(nn.Module):
    def __init__(self, video_dim=512, hidden_dim=512, num_classes=44, max_proposals=10, num_layers=2, num_heads=8):
        super(TransformerActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_proposals = max_proposals

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=video_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )

        # Proposal generator (start, end, confidence)
        self.proposal_head = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * max_proposals)  # 3 outputs per proposal: start, end, confidence
        )

        # Action classifier
        self.classifier = nn.Linear(video_dim, num_classes * max_proposals)

    def forward(self, video):
        # video: (batch_size, max_length, video_dim), e.g., (8, 300, 512)
        # Mask for padded frames (if any)
        mask = (video.sum(dim=-1) != 0).bool()  # True where frames are valid

        # Transformer encoding
        transformer_out = self.transformer(video, src_key_padding_mask=~mask)  # Shape: (batch_size, max_length, video_dim)
        transformer_avg = transformer_out.mean(dim=1)  # Shape: (batch_size, video_dim)

        # Predict proposals
        proposals = self.proposal_head(transformer_avg)  # Shape: (batch_size, 3 * max_proposals)
        proposals = proposals.view(-1, self.max_proposals, 3)  # Shape: (batch_size, max_proposals, 3)

        # Predict class scores
        class_scores = self.classifier(transformer_avg)  # Shape: (batch_size, num_classes * max_proposals)
        class_scores = class_scores.view(-1, self.max_proposals, self.num_classes)  # Fixed: self.num_classes

        # Softmax over classes
        class_probs = torch.softmax(class_scores, dim=-1)

        return proposals, class_probs  # (batch_size, max_proposals, 3), (batch_size, max_proposals, num_classes)