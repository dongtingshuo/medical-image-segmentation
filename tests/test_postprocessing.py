import torch

from src.postprocessing import fill_mask_holes, postprocess_binary_masks, remove_small_components


def test_remove_small_components_drops_tiny_regions():
    mask = torch.zeros(1, 1, 6, 6)
    mask[0, 0, 1, 1] = 1
    mask[0, 0, 3:6, 3:6] = 1

    processed = postprocess_binary_masks(mask, min_component_area=4)

    assert processed.sum().item() == 9
    assert processed[0, 0, 1, 1].item() == 0


def test_fill_mask_holes_preserves_outer_background():
    mask = torch.ones(1, 1, 5, 5)
    mask[0, 0, 2, 2] = 0
    mask[0, 0, 0, 0] = 0

    processed = postprocess_binary_masks(mask, fill_holes=True)

    assert processed[0, 0, 2, 2].item() == 1
    assert processed[0, 0, 0, 0].item() == 0


def test_numpy_helpers_return_float_masks():
    mask = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]

    assert remove_small_components(mask, min_area=2).dtype.name == "float32"
    assert fill_mask_holes(mask).dtype.name == "float32"
