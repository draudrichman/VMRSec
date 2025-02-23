# eval_transformer.py
import torch
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset, load_action_mappings
from models.custom_transformer_model import TransformerActionModel

def compute_iou(pred_segment, gt_segment):
    """Calculate Intersection over Union (IoU) between predicted and ground truth segments."""
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0

def evaluate(model, dataloader, device, iou_threshold=0.1):
    """Evaluate the model on the test dataset."""
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0  # True Positives, False Positives, False Negatives
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)  # Shape: (batch_size, max_length, video_dim)
            actions = batch['actions'].to(device)  # Shape: (batch_size, max_proposals, 3)
            durations = batch['duration'].clone().detach().to(device)  # Shape: (batch_size,)

            # Model prediction
            proposals, class_probs = model(video)  # (batch_size, max_proposals, 3), (batch_size, max_proposals, num_classes)
            batch_size = video.size(0)
            max_length = video.size(1)  # 300
            scale_factors = durations / max_length  # Scale from feature space to seconds

            # Process each video in the batch
            for i in range(batch_size):
                pred_segments = []
                # Extract predicted segments with confidence > 0.1
                for j in range(proposals.size(1)):
                    pred_class = torch.argmax(class_probs[i, j]).item()
                    pred_start, pred_end = proposals[i, j, :2] * scale_factors[i]  # Scale to seconds
                    confidence = class_probs[i, j, pred_class].item()
                    if confidence > 0.1:  # Confidence threshold
                        pred_segments.append({'class': pred_class, 'start': pred_start.item(), 'end': pred_end.item()})

                # Ground truth segments
                gt_segments = []
                for action in actions[i]:
                    class_idx = int(action[0].item())
                    if class_idx != -1:  # Skip padding
                        gt_segments.append({'class': class_idx, 'start': action[1].item(), 'end': action[2].item()})

                # Match predictions to ground truth
                matched_gt = set()
                for pred in pred_segments:
                    best_iou = 0
                    best_gt_idx = -1
                    for gt_idx, gt in enumerate(gt_segments):
                        if gt['class'] == pred['class'] and gt_idx not in matched_gt:
                            iou = compute_iou((pred['start'], pred['end']), (gt['start'], gt['end']))
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx
                    if best_iou >= iou_threshold:  # Correct prediction
                        all_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        all_fp += 1  # False positive

                all_fn += len(gt_segments) - len(matched_gt)  # False negatives (unmatched GT)

    # Calculate precision and recall
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    return precision, recall

def main():
    # Hyperparameters matching your training setup
    video_dim = 512
    hidden_dim = 512
    num_classes = 44
    batch_size = 8
    max_proposals = 10
    max_length = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')
    test_dataset = CustomDataset(
        label_path='data/custom_dataset/test_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        action_mappings=action_mappings,
        num_classes=num_classes,
        max_length=max_length,
        max_proposals=max_proposals
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load trained Transformer model
    model = TransformerActionModel(
        video_dim=video_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        max_proposals=max_proposals
    ).to(device)
    model.load_state_dict(torch.load('transformer_model.pth', weights_only=True))
    print("Loaded trained model from 'transformer_model.pth'")

    # Evaluate
    precision, recall = evaluate(model, test_dataloader, device, iou_threshold=0.1)
    print(f'Precision @ IoU=0.1: {precision:.2f}, Recall @ IoU=0.1: {recall:.2f}')

if __name__ == '__main__':
    main()