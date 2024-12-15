import pickle
from pathlib import Path

import numpy as np
import torch

from src.minimum_spanning_tree import minimum_spanning_tree_v2


def test_minimum_spanning_tree_v2():
    msp_inputs = pickle.load((Path(__file__).parent / "msp_inputs.p").open("rb"))
    msp_outputs = pickle.load((Path(__file__).parent / "msp_outputs.p").open("rb"))
    pts3d, msp_edges, im_focals, im_poses = minimum_spanning_tree_v2(**msp_inputs)

    assert_all(msp_outputs["pts3d"], pts3d)
    assert_all(msp_outputs["msp_edges"], msp_edges)
    assert_all(msp_outputs["im_poses"], im_poses)
    assert_all(msp_outputs["im_focals"], im_focals)


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
