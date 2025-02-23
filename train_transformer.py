import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset, load_action_mappings
from models.custom_transformer_model import TransformerActionModel

def train(model, dataloader, criterion_cls, criterion_reg, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        video = batch['video'].to(device)  # Shape: (batch_size, max_length, video_dim)
        actions = batch['actions'].to(device)  # Shape: (batch_size, max_proposals, 3)
        durations = batch['duration'].clone().detach().to(device)  # Fixed: clone().detach().to()

        optimizer.zero_grad()
        proposals, class_probs = model(video)

        # Ground truth
        gt_classes = actions[:, :, 0].long()
        gt_segments = actions[:, :, 1:]
        batch_size, max_length = video.size(0), video.size(1)
        scale_factors = max_length / durations.unsqueeze(1).unsqueeze(2)
        gt_segments_normalized = gt_segments * scale_factors

        # Loss
        valid_mask = gt_classes != -1
        cls_loss = criterion_cls(class_probs[valid_mask], gt_classes[valid_mask])
        reg_loss = criterion_reg(proposals[:, :, :2][valid_mask], gt_segments_normalized[valid_mask])
        loss = cls_loss + reg_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    video_dim = 512
    hidden_dim = 512
    num_classes = 44
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    max_proposals = 10
    max_length = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')
    train_dataset = CustomDataset(
        label_path='data/custom_dataset/train_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        action_mappings=action_mappings,
        num_classes=num_classes,
        max_length=max_length,
        max_proposals=max_proposals
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TransformerActionModel(
        video_dim=video_dim, 
        hidden_dim=hidden_dim, 
        num_classes=num_classes, 
        max_proposals=max_proposals
    ).to(device)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=-1)
    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion_cls, criterion_reg, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    torch.save(model.state_dict(), 'transformer_model.pth')

if __name__ == '__main__':
    main()