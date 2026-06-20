import cv2

from scripts.create_toy_segmentation_data import create_toy_segmentation_data


def test_create_toy_segmentation_data_generates_matched_images_and_masks(tmp_path):
    images_dir, masks_dir = create_toy_segmentation_data(tmp_path / "toy", num_samples=4, image_size=64)
    images = sorted(images_dir.glob("*.png"))
    masks = sorted(masks_dir.glob("*.png"))
    assert len(images) == 4
    assert [path.name for path in images] == [path.name for path in masks]
    mask = cv2.imread(str(masks[0]), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert set(mask.flatten().tolist()).issubset({0, 255})
