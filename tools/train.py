# tools/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from models.custom_model import CustomModel
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_action_mappings(file_path):
    """
    Load action codes and their meanings from a text file.

    Args:
        file_path (str): Path to the actions.txt file.

    Returns:
        dict: A dictionary mapping action codes to their meanings.
    """
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
        video = batch['video'].to(device)
        actions = batch['actions'].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(video)  # Shape: (batch_size, num_classes)
        
        # Compute loss
        loss = criterion(outputs, actions)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def main():
    # Hyperparameters
    video_dim = 2048  # Example: Dimension of video features
    hidden_dim = 512  # Hidden layer dimension
    num_classes = 44  # Number of action classes (from actions.txt)
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
        num_classes=num_classes,  # Pass num_classes here
        max_length=300  # Set max_length for padding
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = CustomModel(video_dim, hidden_dim, num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'custom_model.pth')

if __name__ == '__main__':
    main()