import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        denominator = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.focal(logits, targets) + 0.5 * self.dice(logits, targets)


def get_loss(loss_name):
    name = str(loss_name).lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss()
    if name == "bce_dice":
        return BCEDiceLoss()
    if name == "focal":
        return FocalLoss()
    if name == "focal_dice":
        return FocalDiceLoss()
    raise ValueError(f"Unsupported loss_name: {loss_name}")


def build_loss(config):
    """Build a binary segmentation loss from a YAML-style config dictionary."""
    loss_cfg = config.get("loss", {}) if isinstance(config, dict) else {}
    training_cfg = config.get("training", {}) if isinstance(config, dict) else {}
    name = loss_cfg.get("name", training_cfg.get("loss_name", config.get("loss_name", "bce_dice")))
    name = str(name).lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return DiceLoss(smooth=float(loss_cfg.get("smooth", 1.0)))
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        )
    if name == "focal":
        return FocalLoss(
            alpha=float(loss_cfg.get("alpha", 0.25)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
        )
    if name == "focal_dice":
        return FocalDiceLoss(
            alpha=float(loss_cfg.get("alpha", 0.25)),
            gamma=float(loss_cfg.get("gamma", 2.0)),
        )
    raise ValueError(f"Unsupported loss name: {name}")
