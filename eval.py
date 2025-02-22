# eval.py
import torch
from torch.utils.data import DataLoader
from datasets.custom_dataset import CustomDataset, load_action_mappings
from models.custom_model import TemporalActionModel

def compute_iou(pred_segment, gt_segment):
    pred_start, pred_end = pred_segment
    gt_start, gt_end = gt_segment
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    return intersection / union if union > 0 else 0

def evaluate(model, dataloader, device, iou_threshold=0.1):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)  # Shape: (batch_size, max_length, video_dim)
            actions = batch['actions'].to(device)  # Shape: (batch_size, max_proposals, 3)

            proposals, class_probs = model(video)  # (batch_size, max_proposals, 3), (batch_size, max_proposals, num_classes)
            batch_size = video.size(0)

            for i in range(batch_size):
                pred_segments = []
                for j in range(proposals.size(1)):
                    pred_class = torch.argmax(class_probs[i, j]).item()
                    pred_start, pred_end = proposals[i, j, :2].tolist()
                    confidence = class_probs[i, j, pred_class].item()
                    if confidence > 0.1:
                        pred_segments.append({'class': pred_class, 'start': pred_start, 'end': pred_end})

                gt_segments = []
                for action in actions[i]:
                    class_idx = int(action[0].item())
                    if class_idx != -1:  # Skip padding
                        gt_segments.append({'class': class_idx, 'start': action[1].item(), 'end': action[2].item()})

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
                    if best_iou >= iou_threshold:
                        all_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        all_fp += 1

                all_fn += len(gt_segments) - len(matched_gt)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    return precision, recall

def main():
    video_dim = 512
    hidden_dim = 512
    num_classes = 44
    batch_size = 8
    max_proposals = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_mappings = load_action_mappings('data/custom_dataset/actions.txt')
    val_dataset = CustomDataset(
        label_path='data/custom_dataset/test_annotations.jsonl',
        video_path='data/custom_dataset/video_features',
        action_mappings=action_mappings,
        num_classes=num_classes,
        max_length=300,
        max_proposals=max_proposals
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TemporalActionModel(video_dim, hidden_dim, num_classes, max_proposals=max_proposals).to(device)
    model.load_state_dict(torch.load('temporal_model.pth', weights_only=True))

    precision, recall = evaluate(model, val_dataloader, device, iou_threshold=0.1)
    print(f'Precision @ IoU=0.1: {precision:.2f}, Recall @ IoU=0.1: {recall:.2f}')

if __name__ == '__main__':
    main()