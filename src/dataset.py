from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _require_albumentations():
    if A is None:
        raise ImportError("Albumentations is required for transforms. Install it with `pip install albumentations`.")


def _as_size(config):
    size = config.get("data", {}).get("image_size", config.get("image_size", 256))
    if isinstance(size, (list, tuple)):
        return int(size[0]), int(size[1])
    return int(size), int(size)


def get_train_transforms(config):
    _require_albumentations()
    height, width = _as_size(config)
    aug_cfg = config.get("augmentation", {})
    transforms = [A.Resize(height=height, width=width)]
    if aug_cfg.get("enabled", True):
        if aug_cfg.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("vertical_flip", False):
            transforms.append(A.VerticalFlip(p=0.5))
        if aug_cfg.get("rotate", True):
            transforms.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5))
        if aug_cfg.get("color_jitter", True):
            transforms.append(A.RandomBrightnessContrast(p=0.3))
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return A.Compose(transforms)


def get_val_transforms(config):
    _require_albumentations()
    height, width = _as_size(config)
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class SkinLesionDataset(Dataset):
    """Skin lesion image/mask dataset with filename-stem matching."""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")

        mask_map = {
            path.stem: path
            for path in self.masks_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        pairs = []
        for image_path in sorted(self.images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_map.get(image_path.stem)
            if mask_path is not None:
                pairs.append((image_path, mask_path))

        if not pairs:
            raise ValueError(
                f"No image/mask pairs found. images_dir={self.images_dir}, masks_dir={self.masks_dir}. "
                "Files must share the same filename stem."
            )
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask

