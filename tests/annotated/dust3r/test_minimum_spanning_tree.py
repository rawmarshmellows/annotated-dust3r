import torch

from src.annotated.dust3r.minimum_spanning_tree import minimum_spanning_tree_v2 as annotated_minimum_spanning_tree
from src.vendored.dust3r.minimum_spanning_tree import minimum_spanning_tree as vendored_minimum_spanning_tree


def test_estimate_focal():
    import pickle
    from pathlib import Path

    msp_inputs = pickle.load(
        (Path(__file__).parent.parent.parent / "minimum_spanning_tree" / "msp_inputs.p").open("rb")
    )

    annotated_pts3d, annotated_msp_edges, annotated_im_focals, annotated_im_poses = annotated_minimum_spanning_tree(
        **msp_inputs
    )
    vendored_pts3d, vendored_msp_edges, vendored_im_focals, vendored_im_poses = vendored_minimum_spanning_tree(
        **msp_inputs
    )

    assert_all(annotated_pts3d, vendored_pts3d)
    assert_all(annotated_msp_edges, vendored_msp_edges)
    assert_all(annotated_im_poses, vendored_im_poses)
    assert_all(annotated_im_focals, vendored_im_focals)


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
