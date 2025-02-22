# tools/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset 
from models.custom_model import CustomModel


def load_action_mappings(file_path):
    action_mappings = {}
    with open(file_path, 'r') as f:
        for line in f:
            code, description = line.strip().split(maxsplit=1)
            action_mappings[code] = description
    return action_mappings

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        video = batch['video'].to(device)  # Shape: (batch_size, max_length, video_dim)
        actions = batch['actions'].to(device)  # Shape: (batch_size, num_classes)

        # Average over frames to match model input (temporary fix)
        video = video.mean(dim=1)  # Shape: (batch_size, video_dim)

        optimizer.zero_grad()
        outputs = model(video)  # Shape: (batch_size, num_classes)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    # Hyperparameters
    video_dim = 512  # Matches CLIP features
    hidden_dim = 512
    num_classes = 44
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load action mappings
    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')

    # Dataset and DataLoader
    train_dataset = CustomDataset(
        label_path='data/custom_dataset/train_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        action_mappings=action_mappings,
        num_classes=num_classes,
        max_length=300
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = CustomModel(video_dim, hidden_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'custom_model.pth')

if __name__ == '__main__':
    main()