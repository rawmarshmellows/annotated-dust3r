import torch


def assert_all(a, b):
    assert isinstance(a, type(b))
    if isinstance(a, torch.Tensor):
        assert torch.all(a == b), f"a: {a}\nb: {b}"
        return

    if isinstance(a, list):
        assert len(a) == len(b)
        for element_a, element_b in zip(a, b):
            assert_all(element_a, element_b)
        return
    assert a == b
    return
