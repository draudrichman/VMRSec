# tools/eval.py
import torch
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from models.custom_model import CustomModel

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            actions = batch['actions'].to(device)
            
            # Forward pass
            outputs = model(video, audio)
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate accuracy
            total += actions.size(0)
            correct += (predicted == actions).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Hyperparameters
    video_dim = 2048  # Example: Dimension of video features
    audio_dim = 128   # Example: Dimension of audio features
    hidden_dim = 512  # Hidden layer dimension
    num_classes = 44  # Number of action classes (from actions.txt)
    batch_size = 8
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset and DataLoader
    val_dataset = CustomDataset(
        label_path='data/custom_dataset/val_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        audio_path='data/custom_dataset/audio_features'
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    model = CustomModel(video_dim, audio_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load('custom_model.pth'))
    
    # Evaluate the model
    accuracy = evaluate(model, val_dataloader, device)
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()