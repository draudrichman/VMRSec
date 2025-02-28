# extract_features.py
import torch
from decord import VideoReader, cpu
from transformers import CLIPProcessor, CLIPModel
import numpy as np

def extract_video_features(video_path, output_path, max_length=300, video_dim=512):
    """
    Extract CLIP features from video frames and save them as .npy file.
    
    Args:
        video_path (str): Path to the .mp4 video file.
        output_path (str): Path to save the extracted .npy features.
        max_length (int): Maximum number of frames to process (padding/truncating if needed).
        video_dim (int): Expected feature dimension (CLIP outputs 512 by default).
    
    Returns:
        tuple: (features, duration, num_frames_extracted)
            - features (np.ndarray): Extracted features of shape (max_length, video_dim).
            - duration (float): Actual video duration in seconds.
            - num_frames_extracted (int): Number of frames extracted before padding/truncating.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load video using decord
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        duration = num_frames / vr.get_avg_fps()  # Actual video duration in seconds
        frame_rate = 1  # Extract 1 frame per second
        frame_indices = range(0, num_frames, int(vr.get_avg_fps() / frame_rate))
        frames = vr.get_batch(frame_indices).asnumpy()
        num_frames_extracted = len(frames)

        # Preprocess frames and move to GPU
        inputs = processor(images=list(frames), return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract features
        with torch.no_grad():
            features = model.get_image_features(**inputs)  # Shape: (num_frames_extracted, 512)
        features = features.cpu().numpy()

        # Pad or truncate to max_length
        if features.shape[0] < max_length:
            padding = np.zeros((max_length - features.shape[0], video_dim))
            features = np.vstack((features, padding))  # Shape: (max_length, video_dim)
        elif features.shape[0] > max_length:
            features = features[:max_length, :]  # Shape: (max_length, video_dim)

        # Save features
        np.save(output_path, features)
        print(f"Saved features to {output_path}. Shape: {features.shape}, Duration: {duration:.2f}s, Frames Extracted: {num_frames_extracted}")
        return features, duration, num_frames_extracted

    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    video_path = "path_to_your_video.mp4"
    output_path = "data/custom_dataset/video_features/new_video.npy"
    features, duration, num_frames = extract_video_features(video_path, output_path)