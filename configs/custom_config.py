# configs/custom_config.py
_base_ = [
    '../_base_/models/custom_model.py',
    '../_base_/datasets/custom_dataset.py',
    '../_base_/schedules/100e.py',
    '../_base_/runtime.py'
]

# Model configuration
model = dict(
    video_dim=2048,
    audio_dim=128,
    hidden_dim=512,
    num_classes=44
)

# Dataset configuration
data = dict(
    train=dict(
        label_path='data/custom_dataset/train_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        audio_path='data/custom_dataset/audio_features'
    ),
    val=dict(
        label_path='data/custom_dataset/val_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        audio_path='data/custom_dataset/audio_features'
    )
)

# Training configuration
stages = dict(
    epochs=10,
    optimizer=dict(type='Adam', lr=0.001),
    validation=dict(interval=1)
)