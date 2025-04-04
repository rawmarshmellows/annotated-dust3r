import pickle
from pathlib import Path

from test_utils import assert_all

from src.annotated.dust3r.minimum_spanning_tree import minimum_spanning_tree_v2 as annotated_minimum_spanning_tree
from src.vendored.dust3r.minimum_spanning_tree import minimum_spanning_tree as vendored_minimum_spanning_tree


def test_minimum_spanning_tree():
    msp_inputs = pickle.load((Path(__file__).parent / "test_data" / "msp_inputs.p").open("rb"))

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
