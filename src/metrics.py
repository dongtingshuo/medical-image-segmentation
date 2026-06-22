import torch
import torch.nn.functional as F


def _binarize(logits, threshold=0.5):
    return (torch.sigmoid(logits) >= threshold).float()


def _flatten(preds, targets):
    return preds.contiguous().view(preds.size(0), -1), targets.contiguous().view(targets.size(0), -1)


def dice_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = _binarize(logits, threshold)
    preds, targets = _flatten(preds, targets.float())
    intersection = (preds * targets).sum(dim=1)
    denominator = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def iou_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = _binarize(logits, threshold)
    preds, targets = _flatten(preds, targets.float())
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def precision_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = _binarize(logits, threshold)
    preds, targets = _flatten(preds, targets.float())
    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    return ((tp + eps) / (tp + fp + eps)).mean()


def recall_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = _binarize(logits, threshold)
    preds, targets = _flatten(preds, targets.float())
    tp = (preds * targets).sum(dim=1)
    fn = ((1.0 - preds) * targets).sum(dim=1)
    return ((tp + eps) / (tp + fn + eps)).mean()


def sensitivity_score(logits, targets, threshold=0.5, eps=1e-7):
    return recall_score(logits, targets, threshold=threshold, eps=eps)


def specificity_score(logits, targets, threshold=0.5, eps=1e-7):
    preds = _binarize(logits, threshold)
    preds, targets = _flatten(preds, targets.float())
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=1)
    fp = (preds * (1.0 - targets)).sum(dim=1)
    return ((tn + eps) / (tn + fp + eps)).mean()


def _mask_boundary(mask):
    mask = (mask > 0.5).float()
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return ((dilated - eroded) > 0).float()


def boundary_f1_score(logits, targets, threshold=0.5, tolerance=2, eps=1e-7):
    preds = _binarize(logits, threshold)
    pred_boundary = _mask_boundary(preds)
    true_boundary = _mask_boundary(targets.float())
    kernel_size = 2 * int(tolerance) + 1
    pred_neighborhood = F.max_pool2d(pred_boundary, kernel_size=kernel_size, stride=1, padding=tolerance)
    true_neighborhood = F.max_pool2d(true_boundary, kernel_size=kernel_size, stride=1, padding=tolerance)

    pred_boundary, true_boundary = _flatten(pred_boundary, true_boundary)
    pred_neighborhood, true_neighborhood = _flatten(pred_neighborhood, true_neighborhood)
    matched_pred = (pred_boundary * true_neighborhood).sum(dim=1)
    matched_true = (true_boundary * pred_neighborhood).sum(dim=1)
    pred_count = pred_boundary.sum(dim=1)
    true_count = true_boundary.sum(dim=1)
    precision = matched_pred / (pred_count + eps)
    recall = matched_true / (true_count + eps)
    score = (2.0 * precision * recall) / (precision + recall + eps)
    score = torch.where((pred_count == 0) & (true_count == 0), torch.ones_like(score), score)
    return score.mean()
