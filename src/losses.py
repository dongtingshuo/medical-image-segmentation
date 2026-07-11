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


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).contiguous().view(logits.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)
        true_positive = (probs * targets).sum(dim=1)
        false_positive = (probs * (1.0 - targets)).sum(dim=1)
        false_negative = ((1.0 - probs) * targets).sum(dim=1)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        return 1.0 - tversky.mean()


def _lovasz_grad(ground_truth_sorted):
    pixels = len(ground_truth_sorted)
    total_positive = ground_truth_sorted.sum()
    intersection = total_positive - ground_truth_sorted.float().cumsum(0)
    union = total_positive + (1.0 - ground_truth_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-7)
    if pixels > 1:
        jaccard = torch.cat((jaccard[:1], jaccard[1:pixels] - jaccard[:-1]))
    return jaccard


class LovaszHingeLoss(nn.Module):
    def forward(self, logits, targets):
        losses = []
        for logit, target in zip(logits, targets):
            logit = logit.reshape(-1)
            target = target.reshape(-1).float()
            signs = 2.0 * target - 1.0
            errors = 1.0 - logit * signs
            errors_sorted, permutation = torch.sort(errors, descending=True)
            target_sorted = target[permutation]
            losses.append(torch.dot(F.relu(errors_sorted), _lovasz_grad(target_sorted)))
        return torch.stack(losses).mean()


class SoftBoundaryDiceLoss(nn.Module):
    def __init__(self, kernel_size=3, smooth=1.0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.smooth = float(smooth)

    def _soft_boundary(self, value):
        padding = self.kernel_size // 2
        dilated = F.max_pool2d(value, kernel_size=self.kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool2d(-value, kernel_size=self.kernel_size, stride=1, padding=padding)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits, targets):
        pred_boundary = self._soft_boundary(torch.sigmoid(logits))
        true_boundary = self._soft_boundary(targets.float())
        pred_boundary = pred_boundary.flatten(1)
        true_boundary = true_boundary.flatten(1)
        intersection = (pred_boundary * true_boundary).sum(dim=1)
        denominator = pred_boundary.sum(dim=1) + true_boundary.sum(dim=1)
        score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - score.mean()


class HybridBoundaryLoss(nn.Module):
    def __init__(self, bce_weight=0.2, dice_weight=0.4, lovasz_weight=0.25, boundary_weight=0.15):
        super().__init__()
        weights = [bce_weight, dice_weight, lovasz_weight, boundary_weight]
        if any(float(weight) < 0 for weight in weights) or sum(float(weight) for weight in weights) <= 0:
            raise ValueError(f"Hybrid loss weights must be non-negative with a positive sum, got {weights}")
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lovasz = LovaszHingeLoss()
        self.boundary = SoftBoundaryDiceLoss()
        self.weights = tuple(float(weight) for weight in weights)

    def forward(self, logits, targets):
        bce_weight, dice_weight, lovasz_weight, boundary_weight = self.weights
        return (
            bce_weight * self.bce(logits, targets)
            + dice_weight * self.dice(logits, targets)
            + lovasz_weight * self.lovasz(logits, targets)
            + boundary_weight * self.boundary(logits, targets)
        )


class DistillationSegmentationLoss(nn.Module):
    def __init__(self, hard_loss=None, hard_weight=0.6, soft_bce_weight=0.25, soft_dice_weight=0.15, temperature=2.0):
        super().__init__()
        self.hard_loss = hard_loss or HybridBoundaryLoss()
        self.hard_weight = float(hard_weight)
        self.soft_bce_weight = float(soft_bce_weight)
        self.soft_dice_weight = float(soft_dice_weight)
        self.temperature = float(temperature)

    def forward(self, logits, targets, soft_targets=None):
        if soft_targets is None:
            raise ValueError("Distillation loss requires OOF soft targets.")
        soft_targets = soft_targets.float().clamp(0.0, 1.0)
        hard = self.hard_loss(logits, targets)
        scaled_logits = logits / self.temperature
        soft_logits = torch.logit(soft_targets.clamp(1e-5, 1.0 - 1e-5)) / self.temperature
        softened_targets = torch.sigmoid(soft_logits)
        soft_bce = F.binary_cross_entropy_with_logits(scaled_logits, softened_targets) * self.temperature**2
        probs = torch.sigmoid(scaled_logits).flatten(1)
        soft_flat = softened_targets.flatten(1)
        soft_dice = 1.0 - ((2.0 * (probs * soft_flat).sum(1) + 1.0) / (probs.sum(1) + soft_flat.sum(1) + 1.0)).mean()
        return self.hard_weight * hard + self.soft_bce_weight * soft_bce + self.soft_dice_weight * soft_dice


class BootstrappedBCEDiceLoss(nn.Module):
    """BCE + Dice with detached self-targets for noisy pretraining labels."""

    def __init__(self, beta=0.85, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        if not 0.0 < float(beta) <= 1.0:
            raise ValueError(f"beta must be in (0, 1], got {beta}")
        if float(bce_weight) < 0 or float(dice_weight) < 0 or float(bce_weight) + float(dice_weight) <= 0:
            raise ValueError("Bootstrapped BCE/Dice weights must be non-negative with a positive total.")
        self.beta = float(beta)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        bootstrap_targets = self.beta * targets + (1.0 - self.beta) * torch.sigmoid(logits).detach()
        bce = F.binary_cross_entropy_with_logits(logits, bootstrap_targets)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class ConfidenceGatedDistillationLoss(nn.Module):
    """Keep hard labels dominant and distil only high-confidence teacher pixels."""

    def __init__(self, hard_loss=None, hard_weight=0.90, soft_bce_weight=0.07, soft_dice_weight=0.03, temperature=2.0, confidence_threshold=0.60):
        super().__init__()
        total = float(hard_weight) + float(soft_bce_weight) + float(soft_dice_weight)
        if total <= 0:
            raise ValueError("Confidence-gated distillation weights must have a positive total.")
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        self.hard_loss = hard_loss or BCEDiceLoss()
        self.hard_weight = float(hard_weight)
        self.soft_bce_weight = float(soft_bce_weight)
        self.soft_dice_weight = float(soft_dice_weight)
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)

    def forward(self, logits, targets, soft_targets):
        hard = self.hard_loss(logits, targets)
        confidence = (soft_targets - 0.5).abs() * 2.0
        gate = (confidence >= self.confidence_threshold).float()
        temperature = max(self.temperature, 1e-6)
        soft_bce = F.binary_cross_entropy_with_logits(logits / temperature, soft_targets, reduction="none")
        soft_bce = (soft_bce * gate).sum() / gate.sum().clamp_min(1.0)
        probabilities = torch.sigmoid(logits / temperature)
        intersection = (probabilities * soft_targets * gate).sum(dim=(1, 2, 3))
        denominator = ((probabilities + soft_targets) * gate).sum(dim=(1, 2, 3))
        soft_dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return self.hard_weight * hard + self.soft_bce_weight * soft_bce + self.soft_dice_weight * soft_dice


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
    if name == "tversky":
        return TverskyLoss()
    if name == "hybrid_boundary":
        return HybridBoundaryLoss()
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
    if name == "tversky":
        return TverskyLoss(
            alpha=float(loss_cfg.get("alpha", 0.3)),
            beta=float(loss_cfg.get("beta", 0.7)),
            smooth=float(loss_cfg.get("smooth", 1.0)),
        )
    if name == "hybrid_boundary":
        return HybridBoundaryLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.20)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.40)),
            lovasz_weight=float(loss_cfg.get("lovasz_weight", 0.25)),
            boundary_weight=float(loss_cfg.get("boundary_weight", 0.15)),
        )
    if name == "distillation":
        hard_cfg = dict(loss_cfg.get("hard_loss", {}))
        hard_loss = HybridBoundaryLoss(
            bce_weight=float(hard_cfg.get("bce_weight", 0.20)),
            dice_weight=float(hard_cfg.get("dice_weight", 0.40)),
            lovasz_weight=float(hard_cfg.get("lovasz_weight", 0.25)),
            boundary_weight=float(hard_cfg.get("boundary_weight", 0.15)),
        )
        return DistillationSegmentationLoss(
            hard_loss=hard_loss,
            hard_weight=float(loss_cfg.get("hard_weight", 0.60)),
            soft_bce_weight=float(loss_cfg.get("soft_bce_weight", 0.25)),
            soft_dice_weight=float(loss_cfg.get("soft_dice_weight", 0.15)),
            temperature=float(loss_cfg.get("temperature", 2.0)),
        )
    if name == "bootstrapped_bce_dice":
        return BootstrappedBCEDiceLoss(
            beta=float(loss_cfg.get("beta", 0.85)),
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        )
    if name == "confidence_gated_distillation":
        hard_cfg = dict(loss_cfg.get("hard_loss", {}))
        hard_loss = BCEDiceLoss(
            bce_weight=float(hard_cfg.get("bce_weight", 0.5)),
            dice_weight=float(hard_cfg.get("dice_weight", 0.5)),
        )
        return ConfidenceGatedDistillationLoss(
            hard_loss=hard_loss,
            hard_weight=float(loss_cfg.get("hard_weight", 0.90)),
            soft_bce_weight=float(loss_cfg.get("soft_bce_weight", 0.07)),
            soft_dice_weight=float(loss_cfg.get("soft_dice_weight", 0.03)),
            temperature=float(loss_cfg.get("temperature", 2.0)),
            confidence_threshold=float(loss_cfg.get("confidence_threshold", 0.60)),
        )
    raise ValueError(f"Unsupported loss name: {name}")
