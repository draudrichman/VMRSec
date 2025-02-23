# datasets/custom_dataset.py
import torch
from torch.utils.data import Dataset
import json
import numpy as np

def load_action_mappings(file_path):
    action_mappings = {}
    with open(file_path, 'r') as f:
        for line in f:
            code, description = line.strip().split(maxsplit=1)
            action_mappings[code] = description
    return action_mappings

class CustomDataset(Dataset):
    def __init__(self, label_path, video_path, action_mappings, num_classes, max_length=300, max_proposals=10):
        with open(label_path, 'r') as f:
            self.label = [json.loads(line) for line in f]
        self.video_path = video_path
        self.action_mappings = action_mappings
        self.num_classes = num_classes
        self.max_length = max_length
        self.max_proposals = max_proposals

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video_id = self.label[idx]['video_id']
        video = np.load(f'{self.video_path}/{video_id}.npy')  # Shape: (num_frames, video_dim)
        
        # Pad or truncate video features
        if video.shape[0] < self.max_length:
            padding = np.zeros((self.max_length - video.shape[0], video.shape[1]))
            video = np.vstack((video, padding))
        elif video.shape[0] > self.max_length:
            video = video[:self.max_length, :]

        # Parse actions with timestamps and pad to max_proposals
        actions_str = self.label[idx]['actions']  # e.g., "c015 0.14 5.96;c016 7.75 12.63"
        action_data = []
        for action in actions_str.split(';'):
            parts = action.split()
            action_code = parts[0]
            start, end = float(parts[1]), float(parts[2])
            action_idx = int(action_code[1:]) - 1  # Zero-based index
            action_data.append([action_idx, start, end])

        # Pad action_data to max_proposals
        num_actions = len(action_data)
        if num_actions < self.max_proposals:
            padding = [[-1, 0.0, 0.0]] * (self.max_proposals - num_actions)
            action_data.extend(padding)
        elif num_actions > self.max_proposals:
            action_data = action_data[:self.max_proposals]

        # Convert to tensors
        video = torch.from_numpy(video).float()  # Shape: (max_length, video_dim)
        actions_tensor = torch.tensor(action_data, dtype=torch.float)  # Shape: (max_proposals, 3)

        return {
            'video': video,
            'actions': actions_tensor,
            'description': self.label[idx]['description'],
            'duration': float(self.label[idx]['duration'])  # Convert duration to float
        }