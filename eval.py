# tools/eval.py
import torch
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset
from models.custom_model import CustomModel

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)  # Shape: (batch_size, max_length, video_dim)
            actions = batch['actions'].to(device)  # Shape: (batch_size, num_classes)
            
            # Average over frames to match model input
            video = video.mean(dim=1)  # Shape: (batch_size, video_dim)
            
            # Forward pass
            outputs = model(video)  # Shape: (batch_size, num_classes)
            
            # For multi-label: Apply sigmoid and threshold
            preds = torch.sigmoid(outputs) > 0.5  # Shape: (batch_size, num_classes)
            
            all_preds.append(preds.cpu())
            all_labels.append(actions.cpu())
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # Shape: (total_samples, num_classes)
    all_labels = torch.cat(all_labels, dim=0)  # Shape: (total_samples, num_classes)
    
    # Calculate accuracy (e.g., mean accuracy across all samples and classes)
    correct = (all_preds == all_labels).float().sum()
    total = all_labels.numel()  # Total number of elements (samples * classes)
    accuracy = 100 * correct / total
    return accuracy.item()

def main():
    # Hyperparameters
    video_dim = 512  # Matches your CLIP features
    hidden_dim = 512
    num_classes = 44
    batch_size = 8
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load action mappings
    def load_action_mappings(file_path):
        action_mappings = {}
        with open(file_path, 'r') as f:
            for line in f:
                code, description = line.strip().split(maxsplit=1)
                action_mappings[code] = description
        return action_mappings
    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')
    
    # Dataset and DataLoader
    val_dataset = CustomDataset(
        label_path='data/custom_dataset/test_annotations.jsonl',  # Using test as val
        video_path='data/custom_dataset/video_features',
        action_mappings=action_mappings,
        num_classes=num_classes,
        max_length=300
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Load the trained model
    model = CustomModel(video_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load('custom_model.pth', weights_only=True))
    
    # Evaluate the model
    accuracy = evaluate(model, val_dataloader, device)
    print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()