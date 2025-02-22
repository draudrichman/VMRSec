# datasets/custom_dataset.py
import torch
from torch.utils.data import Dataset
import json
import numpy as np

# Function to load action mappings from actions.txt
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

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, label_path, video_path, action_mappings, num_classes, transform=None, max_length=300):
        """
        Args:
            label_path (str): Path to the annotation file.
            video_path (str): Path to the directory containing video features.
            action_mappings (dict): Dictionary mapping action codes to their meanings.
            num_classes (int): Number of action classes.
            transform (callable, optional): Optional transform to be applied on a sample.
            max_length (int): Maximum length to pad video features.
        """
        # Load annotations
        with open(label_path, 'r') as f:
            self.label = [json.loads(line) for line in f]

        self.video_path = video_path
        self.action_mappings = action_mappings
        self.num_classes = num_classes
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Load video features
        video_id = self.label[idx]['video_id']
        video = np.load(f'{self.video_path}/{video_id}.npy')  # Video features

        # Pad video features to max_length
        if video.shape[0] < self.max_length:
            padding = np.zeros((self.max_length - video.shape[0], video.shape[1]))
            video = np.vstack((video, padding))
        elif video.shape[0] > self.max_length:
            video = video[:self.max_length, :]

        # Load annotations
        actions_str = self.label[idx]['actions']  # Actions as a string (e.g., "c015 0.14 5.96;c016 7.75 12.63")
        description = self.label[idx]['description']  # Textual description (optional)

        # Convert actions string to a binary vector
        action_labels = torch.zeros(self.num_classes, dtype=torch.float)  # Binary vector
        for action in actions_str.split(';'):
            code = action.split()[0]  # Extract the action code (e.g., "c015")
            action_idx = int(code[1:]) - 1  # Convert "c015" to 14 (zero-based index)
            action_labels[action_idx] = 1.0  # Set the corresponding index to 1

        # Convert to tensors
        video = torch.from_numpy(video).float()

        # Apply transformations (if any)
        if self.transform:
            video = self.transform(video)

        # Return sample
        sample = {
            'video': video,
            'actions': action_labels,  # Binary vector for multi-label classification
            'description': description,
            'action_mappings': self.action_mappings  # Include action mappings
        }
        return sample