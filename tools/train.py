# tools/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from models.custom_model import CustomModel

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        actions = batch['actions'].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(video, audio)
        loss = criterion(outputs, actions)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

def main():
    # Hyperparameters
    video_dim = 2048  # Example: Dimension of video features
    audio_dim = 128   # Example: Dimension of audio features
    hidden_dim = 512  # Hidden layer dimension
    num_classes = 44  # Number of action classes (from actions.txt)
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    train_dataset = CustomDataset(
        label_path='data/custom_dataset/train_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        audio_path='data/custom_dataset/audio_features'
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model, Loss, Optimizer
    model = CustomModel(video_dim, audio_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), 'custom_model.pth')

if __name__ == '__main__':
    main()