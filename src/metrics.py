import torch


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

