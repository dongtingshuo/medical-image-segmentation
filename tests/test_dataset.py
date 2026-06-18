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
