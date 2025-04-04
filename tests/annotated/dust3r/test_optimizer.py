from copy import deepcopy

import pytest
import torch
from loguru import logger

from src.annotated.dust3r.optimizer import PointCloudOptimizer as AnnotatedPointCloudOptimizer
from src.vendored.dust3r.optimizer import PointCloudOptimizer


@pytest.mark.skip
def test_optimizer_forward_pass(model_predictions):
    """Test initialization from 3D points using cached predictions."""
    # Debug: Using cached predictions from fixture
    view1 = model_predictions["view1"]
    view2 = model_predictions["view2"]
    pred1 = model_predictions["pred1"]
    pred2 = model_predictions["pred2"]

    # Verify predictions contain expected keys
    assert "pts3d" in pred1, "pts3d missing from pred1"
    assert "conf" in pred1, "conf missing from pred1"
    assert "pts3d_in_other_view" in pred2, "pts3d_in_other_view missing from pred2"
    assert "conf" in pred2, "conf missing from pred2"

    # Initialize optimizer with cached predictions
    torch.manual_seed(42)
    optimizer = PointCloudOptimizer(view1, view2, pred1, pred2, device=torch.device("cpu"))
    torch.manual_seed(42)
    annotated_optimizer = AnnotatedPointCloudOptimizer(view1, view2, pred1, pred2, device=torch.device("cpu"))

    output = optimizer()
    annotated_output = annotated_optimizer()

    assert torch.allclose(output, annotated_output)
    assert torch.allclose(optimizer.get_pw_poses(), annotated_optimizer.pw_poses)
    assert torch.allclose(optimizer.get_pw_scale(), annotated_optimizer.pw_scale)
    assert torch.allclose(optimizer.get_im_poses(), annotated_optimizer.im_poses)
    assert torch.allclose(optimizer.get_focals(), annotated_optimizer.focals)
    assert torch.allclose(optimizer.get_principal_points(), annotated_optimizer.principal_points)


def test_optimizer_compute_global_alignment(model_predictions):
    """Test initialization from 3D points using cached predictions."""

    # Initialize optimizer with cached predictions
    view1 = model_predictions["view1"]
    view2 = model_predictions["view2"]
    pred1 = model_predictions["pred1"]
    pred2 = model_predictions["pred2"]
    optimizer = PointCloudOptimizer(view1, view2, pred1, pred2, device=torch.device("cpu"), optimize_pp=True)
    optimizer.compute_global_alignment(init="mst", niter=5)

    annotated_optimizer = AnnotatedPointCloudOptimizer(
        view1, view2, pred1, pred2, device=torch.device("cpu"), optimize_principle_points=True
    )
    annotated_optimizer.compute_global_alignment_v2(init="mst", niter=5)

    assert torch.allclose(optimizer.get_pw_poses(), annotated_optimizer.pw_poses)
    assert torch.allclose(optimizer.get_pw_scale(), annotated_optimizer.pw_scale)
    assert torch.allclose(optimizer.get_im_poses(), annotated_optimizer.im_poses)
    assert torch.allclose(optimizer.get_focals(), annotated_optimizer.focals)
    assert torch.allclose(optimizer.get_principal_points(), annotated_optimizer.principal_points)
