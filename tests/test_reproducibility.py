import torch

from src.utils import make_torch_generator, set_seed


def test_seed_and_generators_are_repeatable():
    set_seed(123, deterministic=True)
    first = torch.rand(5)
    set_seed(123, deterministic=True)
    second = torch.rand(5)
    assert torch.equal(first, second)
    assert torch.are_deterministic_algorithms_enabled()

    first_generator = make_torch_generator(99)
    second_generator = make_torch_generator(99)
    assert torch.equal(torch.randperm(20, generator=first_generator), torch.randperm(20, generator=second_generator))
