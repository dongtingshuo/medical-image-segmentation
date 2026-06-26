from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def remove_small_components(mask, min_area=0):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    min_area = int(min_area or 0)
    if min_area <= 0:
        return mask.astype(np.float32)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(np.float32)


def fill_mask_holes(mask):
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    inverted = 1 - mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if count <= 1:
        return mask.astype(np.float32)
    border_labels = set(np.unique(labels[0, :]))
    border_labels.update(np.unique(labels[-1, :]))
    border_labels.update(np.unique(labels[:, 0]))
    border_labels.update(np.unique(labels[:, -1]))
    filled = mask.copy()
    for label in range(1, count):
        if label not in border_labels and int(stats[label, cv2.CC_STAT_AREA]) > 0:
            filled[labels == label] = 1
    return filled.astype(np.float32)


def postprocess_binary_masks(masks, min_component_area=0, fill_holes=False):
    masks_tensor = torch.as_tensor(masks)
    original_device = masks_tensor.device
    original_dtype = masks_tensor.dtype
    array = masks_tensor.detach().cpu().numpy()
    if array.ndim != 4:
        raise ValueError(f"Expected masks with shape (N, C, H, W), got {array.shape}")
    processed = np.zeros_like(array, dtype=np.float32)
    for batch_index in range(array.shape[0]):
        for channel_index in range(array.shape[1]):
            mask = array[batch_index, channel_index]
            if fill_holes:
                mask = fill_mask_holes(mask)
            mask = remove_small_components(mask, min_area=min_component_area)
            processed[batch_index, channel_index] = mask
    return torch.as_tensor(processed, device=original_device, dtype=original_dtype)


def _resize_batch(images, size):
    return F.interpolate(images, size=size, mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_probabilities_tta(model, images, scales=(1.0,), horizontal_flip=False, vertical_flip=False):
    if not scales:
        scales = (1.0,)
    original_size = images.shape[-2:]
    transforms = [(False, False)]
    if horizontal_flip:
        transforms.append((True, False))
    if vertical_flip:
        transforms.append((False, True))
    if horizontal_flip and vertical_flip:
        transforms.append((True, True))

    probabilities = []
    for scale in scales:
        scale = float(scale)
        if scale <= 0:
            raise ValueError(f"TTA scale must be positive, got {scale}")
        if scale == 1.0:
            scaled_images = images
        else:
            scaled_size = (
                max(1, int(round(original_size[0] * scale))),
                max(1, int(round(original_size[1] * scale))),
            )
            scaled_images = _resize_batch(images, scaled_size)
        for flip_h, flip_v in transforms:
            augmented = scaled_images
            if flip_h:
                augmented = torch.flip(augmented, dims=(-1,))
            if flip_v:
                augmented = torch.flip(augmented, dims=(-2,))
            probs = torch.sigmoid(model(augmented))
            if flip_v:
                probs = torch.flip(probs, dims=(-2,))
            if flip_h:
                probs = torch.flip(probs, dims=(-1,))
            if probs.shape[-2:] != original_size:
                probs = _resize_batch(probs, original_size)
            probabilities.append(probs)
    return torch.stack(probabilities, dim=0).mean(dim=0)
