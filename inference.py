# inference.py
import torch
import numpy as np
from models.custom_model import TemporalActionModel
from datasets.custom_dataset import load_action_mappings
from extract_features import extract_video_features

def infer_action_segments(video_path, action_code, model_path='temporal_model.pth', video_dim=512, hidden_dim=512, num_classes=44, max_proposals=10, max_length=300):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load action mappings
    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')
    action_idx = int(action_code[1:]) - 1  # Convert "c001" to 0-based index

    # Extract features from the video
    output_feature_path = 'data/custom_dataset/video_features/temp_video.npy'
    features = extract_video_features(video_path, output_feature_path, max_length=max_length, video_dim=video_dim)
    if features is None:
        print("Feature extraction failed. Exiting.")
        return []

    video_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)  # Shape: (1, max_length, video_dim)

    # Load the trained model
    model = TemporalActionModel(video_dim, hidden_dim, num_classes, max_proposals=max_proposals).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Inference
    with torch.no_grad():
        proposals, class_probs = model(video_tensor)  # Shape: (1, max_proposals, 3), (1, max_proposals, num_classes)
        proposals = proposals.squeeze(0)  # Shape: (max_proposals, 3)
        class_probs = class_probs.squeeze(0)  # Shape: (max_proposals, num_classes)

        # Find segments for the specified action
        segments = []
        for i in range(max_proposals):
            pred_class = torch.argmax(class_probs[i]).item()
            confidence = class_probs[i, pred_class].item()
            if pred_class == action_idx and confidence > 0.5:  # Confidence threshold
                start, end = proposals[i, :2].tolist()
                segments.append((start, end))

    return segments

def main():
    video_path = "V019.mp4"  # Replace with your video path
    action_code = "c036"  # Example: Stealing a parcel
    segments = infer_action_segments(video_path, action_code)
    
    print(f"Action '{action_code}' segments found:")
    for start, end in segments:
        print(f"Start: {start:.2f}s, End: {end:.2f}s")

if __name__ == "__main__":
    main()