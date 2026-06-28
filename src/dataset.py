from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
except ImportError:  # pragma: no cover
    A = None


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}


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
    resize_mode = str(config.get("data", {}).get("resize_mode", "stretch")).lower()
    if resize_mode == "letterbox":
        transforms = [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    elif resize_mode == "stretch":
        transforms = [A.Resize(height=height, width=width)]
    else:
        raise ValueError(f"Unsupported data.resize_mode: {resize_mode}")
    if aug_cfg.get("enabled", True):
        if aug_cfg.get("horizontal_flip", True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if aug_cfg.get("vertical_flip", False):
            transforms.append(A.VerticalFlip(p=0.5))
        controlled = str(aug_cfg.get("strategy", "legacy")).lower() == "controlled"
        if controlled:
            transforms.append(
                A.OneOf(
                    [
                        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=25, p=1.0),
                        A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
                    ],
                    p=float(aug_cfg.get("geometry_p", 0.55)),
                )
            )
            transforms.append(
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=1.0),
                        A.RandomGamma(gamma_limit=(75, 125), p=1.0),
                        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=1.0),
                        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=10, p=1.0),
                    ],
                    p=float(aug_cfg.get("photometric_p", 0.65)),
                )
            )
            transforms.append(
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                        A.CoarseDropout(max_holes=8, max_height=24, max_width=24, min_holes=1, p=1.0),
                    ],
                    p=float(aug_cfg.get("artifact_p", 0.20)),
                )
            )
        else:
            if aug_cfg.get("rotate", True):
                transforms.append(A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5))
            if aug_cfg.get("color_jitter", True):
                transforms.append(A.RandomBrightnessContrast(p=0.3))
        low_contrast_cfg = aug_cfg.get("low_contrast", {})
        if low_contrast_cfg.get("enabled", False) and not controlled:
            if low_contrast_cfg.get("clahe", True):
                transforms.append(
                    A.CLAHE(
                        clip_limit=tuple(low_contrast_cfg.get("clahe_clip_limit", (1.0, 4.0))),
                        tile_grid_size=tuple(low_contrast_cfg.get("clahe_tile_grid_size", (8, 8))),
                        p=float(low_contrast_cfg.get("clahe_p", 0.35)),
                    )
                )
            if low_contrast_cfg.get("random_gamma", True):
                transforms.append(
                    A.RandomGamma(
                        gamma_limit=tuple(low_contrast_cfg.get("gamma_limit", (70, 130))),
                        p=float(low_contrast_cfg.get("gamma_p", 0.35)),
                    )
                )
            if low_contrast_cfg.get("brightness_contrast", True):
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=float(low_contrast_cfg.get("brightness_limit", 0.18)),
                        contrast_limit=float(low_contrast_cfg.get("contrast_limit", 0.35)),
                        p=float(low_contrast_cfg.get("brightness_contrast_p", 0.45)),
                    )
                )
            if low_contrast_cfg.get("simulate_low_contrast", True):
                transforms.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=tuple(low_contrast_cfg.get("simulate_brightness_limit", (-0.05, 0.05))),
                        contrast_limit=tuple(low_contrast_cfg.get("simulate_contrast_limit", (-0.50, -0.15))),
                        p=float(low_contrast_cfg.get("simulate_p", 0.35)),
                    )
                )
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    additional_targets = {"soft_mask": "mask"} if config.get("data", {}).get("soft_masks_dir") else {}
    compose = A.Compose(transforms, additional_targets=additional_targets)
    if hasattr(compose, "set_random_seed"):
        compose.set_random_seed(int(config.get("seed", 42)))
    return compose


def get_val_transforms(config):
    _require_albumentations()
    height, width = _as_size(config)
    resize_mode = str(config.get("data", {}).get("resize_mode", "stretch")).lower()
    if resize_mode == "letterbox":
        transforms = [
            A.LongestMaxSize(max_size=max(height, width)),
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
        ]
    elif resize_mode == "stretch":
        transforms = [A.Resize(height=height, width=width)]
    else:
        raise ValueError(f"Unsupported data.resize_mode: {resize_mode}")
    transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    compose = A.Compose(transforms)
    if hasattr(compose, "set_random_seed"):
        compose.set_random_seed(int(config.get("seed", 42)) + 1)
    return compose


class SkinLesionDataset(Dataset):
    """Skin lesion image/mask dataset with strict filename-stem matching."""

    def __init__(self, images_dir, masks_dir, transform=None, strict=True, soft_masks_dir=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.strict = strict
        self.soft_masks_dir = Path(soft_masks_dir) if soft_masks_dir else None

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory does not exist: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory does not exist: {self.masks_dir}")

        image_paths = [
            path
            for path in sorted(self.images_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        mask_paths = [
            path
            for path in sorted(self.masks_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        mask_map = {
            path.stem: path
            for path in mask_paths
        }
        image_stems = {path.stem for path in image_paths}
        missing_masks = []
        pairs = []
        for image_path in image_paths:
            mask_path = mask_map.get(image_path.stem)
            if mask_path is not None:
                pairs.append((image_path, mask_path))
            else:
                missing_masks.append(image_path.name)

        extra_masks = sorted(set(mask_map) - image_stems)
        if self.strict and (len(image_paths) != len(mask_paths) or missing_masks or extra_masks):
            details = [
                f"Image/mask files do not match exactly. images={len(image_paths)}, masks={len(mask_paths)}.",
                f"images_dir={self.images_dir}",
                f"masks_dir={self.masks_dir}",
            ]
            if missing_masks:
                details.append(f"Missing masks for images: {missing_masks[:10]}")
            if extra_masks:
                details.append(f"Extra masks without images: {extra_masks[:10]}")
            raise ValueError(" ".join(details))

        if not pairs:
            raise ValueError(
                f"No image/mask pairs found. images_dir={self.images_dir}, masks_dir={self.masks_dir}. "
                "Files must share the same filename stem."
            )
        self.pairs = pairs
        self.soft_mask_map = {}
        if self.soft_masks_dir is not None:
            if not self.soft_masks_dir.exists():
                raise FileNotFoundError(f"Soft-mask directory does not exist: {self.soft_masks_dir}")
            self.soft_mask_map = {
                path.stem: path
                for path in self.soft_masks_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            missing_soft = [image_path.stem for image_path, _ in pairs if image_path.stem not in self.soft_mask_map]
            if missing_soft:
                raise ValueError(f"Missing distillation soft masks for: {missing_soft[:10]}")

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
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size mismatch for `{image_path.name}` and `{mask_path.name}`: "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}. Fix the dataset before training."
            )
        mask = (mask > 127).astype(np.float32)

        soft_mask = None
        if self.soft_masks_dir is not None:
            soft_path = self.soft_mask_map[image_path.stem]
            soft_mask = cv2.imread(str(soft_path), cv2.IMREAD_UNCHANGED)
            if soft_mask is None:
                raise ValueError(f"Failed to read soft mask: {soft_path}")
            soft_mask = soft_mask.astype(np.float32) / float(np.iinfo(soft_mask.dtype).max)
            if soft_mask.shape[:2] != mask.shape[:2]:
                raise ValueError(f"Soft mask size mismatch for `{image_path.stem}`")

        if self.transform is not None:
            inputs = {"image": image, "mask": mask}
            if soft_mask is not None:
                inputs["soft_mask"] = soft_mask
            augmented = self.transform(**inputs)
            image = augmented["image"]
            mask = augmented["mask"]
            soft_mask = augmented.get("soft_mask", soft_mask)
        else:
            image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        if soft_mask is None:
            return image, mask
        soft_mask = torch.from_numpy(np.asarray(soft_mask)).float().unsqueeze(0)
        return image, mask, soft_mask
