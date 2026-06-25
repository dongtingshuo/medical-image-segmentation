import numpy as np
import pytest


def test_dataset_strict_matching_rejects_missing_mask(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.dataset import SkinLesionDataset

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "a.jpg"), image)
    cv2.imwrite(str(images_dir / "b.jpg"), image)
    cv2.imwrite(str(masks_dir / "a.png"), mask)

    with pytest.raises(ValueError, match="do not match exactly"):
        SkinLesionDataset(images_dir, masks_dir)


def test_dataset_rejects_image_mask_size_mismatch(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from src.dataset import SkinLesionDataset

    images_dir = tmp_path / "images"
    masks_dir = tmp_path / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "a.jpg"), image)
    cv2.imwrite(str(masks_dir / "a.png"), mask)

    dataset = SkinLesionDataset(images_dir, masks_dir, transform=None)
    with pytest.raises(ValueError, match="size mismatch"):
        _ = dataset[0]


def test_low_contrast_train_transform_preserves_shapes_and_binary_mask():
    pytest.importorskip("albumentations")
    from src.dataset import get_train_transforms

    config = {
        "seed": 42,
        "data": {"image_size": 64},
        "augmentation": {
            "enabled": True,
            "horizontal_flip": True,
            "vertical_flip": True,
            "rotate": True,
            "color_jitter": True,
            "low_contrast": {
                "enabled": True,
                "clahe": True,
                "clahe_p": 1.0,
                "random_gamma": True,
                "gamma_p": 1.0,
                "brightness_contrast": True,
                "brightness_contrast_p": 1.0,
                "simulate_low_contrast": True,
                "simulate_p": 1.0,
            },
        },
    }
    image = np.full((80, 72, 3), 128, dtype=np.uint8)
    mask = np.zeros((80, 72), dtype=np.float32)
    mask[20:50, 20:45] = 1.0

    transformed = get_train_transforms(config)(image=image, mask=mask)

    assert transformed["image"].shape == (64, 64, 3)
    assert transformed["mask"].shape == (64, 64)
    assert set(np.unique(transformed["mask"]).tolist()).issubset({0.0, 1.0})
