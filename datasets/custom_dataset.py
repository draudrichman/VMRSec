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
    def __init__(self, label_path, video_path, audio_path, action_mappings, transform=None):
        """
        Args:
            label_path (str): Path to the annotation file.
            video_path (str): Path to the directory containing video features.
            audio_path (str): Path to the directory containing audio features.
            action_mappings (dict): Dictionary mapping action codes to their meanings.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Load annotations
        with open(label_path, 'r') as f:
            self.label = [json.loads(line) for line in f]
        
        self.video_path = video_path
        self.audio_path = audio_path
        self.action_mappings = action_mappings
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Load video and audio features
        video_id = self.label[idx]['video_id']
        video = np.load(f'{self.video_path}/{video_id}.npy')  # Video features
        audio = np.load(f'{self.audio_path}/{video_id}.npy')  # Audio features

        # Load annotations
        actions = self.label[idx]['actions']  # List of actions with timestamps
        description = self.label[idx]['description']  # Textual description (optional)

        # Convert to tensors
        video = torch.from_numpy(video).float()
        audio = torch.from_numpy(audio).float()

        # Apply transformations (if any)
        if self.transform:
            video = self.transform(video)
            audio = self.transform(audio)

        # Return sample
        sample = {
            'video': video,
            'audio': audio,
            'actions': actions,
            'description': description,
            'action_mappings': self.action_mappings  # Include action mappings
        }
        return sample