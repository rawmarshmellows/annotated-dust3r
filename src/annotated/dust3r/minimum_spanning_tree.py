from dataclasses import dataclass

import cv2
import numpy as np
import roma
import scipy.sparse as sp
import torch


def minimum_spanning_tree_v2(
    imshapes,
    edges,
    pred_source,
    pred_target,
    conf_source,
    conf_target,
    im_conf,
    min_conf_thr,
    device,
    has_im_poses=True,
    niter_PnP=10,
):
    """
    1. Calculate the mean edge score, computed from the average of each pixel in image
    2. Construct the minimum spanning tree of the negated edge scores (which gives us global scores)
    3. Construct the global coordinate point map
    4. Calculate the focals for each image
    5. Calculate the global poses for each image using PnP

    Args:
        imshapes: List of image shapes
        edges: List of edges between images
        pred_source: Dictionary mapping edge strings to source predictions
        pred_target: Dictionary mapping edge strings to target predictions
        conf_source: Dictionary mapping edge strings to source confidences
        conf_target: Dictionary mapping edge strings to target confidences
        im_conf: Image confidences
        min_conf_thr: Minimum confidence threshold
        device: Torch device
        has_im_poses: Whether to compute image poses
        niter_PnP: Number of PnP iterations

    Returns:
        pts3d: List of 3D points in world coordinates
        msp_edges: List of edges in the maximum spanning tree
        im_focals: List of focal lengths for each image
        im_poses: List of poses for each image
    """

    # Step 1: Calculate edge scores and build the maximum spanning tree
    n_imgs = len(imshapes)
    edge_to_edge_confidence_score = compute_edge_scores(edges, pred_source, pred_target, conf_source, conf_target)
    sparse_graph = construct_sparse_graph_ordered_by_negative_confidence_score(edge_to_edge_confidence_score, n_imgs)
    maximum_spanning_tree_edges = extract_maximum_spanning_tree_edges(sparse_graph)

    # Initialize data structures
    pts3d = [None] * len(imshapes)  # 3D points in world coordinates
    im_poses = [None] * n_imgs  # Camera poses in world coordinates
    im_focals = [None] * n_imgs  # Focal lengths for each image
    msp_edges = []  # Edges in the maximum spanning tree
    done = set()  # Set of vertices already processed

    # Step 2: Initialize with the strongest edge
    strongest_edge = maximum_spanning_tree_edges.pop(0)
    pts3d, im_poses, im_focals, done, msp_edges = initialize_tree_with_strongest_edge(
        strongest_edge, pts3d, im_poses, im_focals, done, msp_edges, pred_source, pred_target, has_im_poses, device
    )

    # Step 3: Grow the tree by adding edges one by one
    pts3d, im_poses, im_focals, done, msp_edges = grow_tree(
        maximum_spanning_tree_edges.copy(),
        pts3d,
        im_poses,
        im_focals,
        done,
        msp_edges,
        pred_source,
        pred_target,
        conf_source,
        conf_target,
        has_im_poses,
        device,
    )

    # Step 4: Complete missing information (focal lengths and poses)
    # This step is crucial for creating a coherent 3D reconstruction
    if has_im_poses:
        im_poses, im_focals = complete_missing_information(
            pts3d,
            im_poses,
            im_focals,
            sparse_graph,
            pred_source,
            im_conf,
            min_conf_thr,
            device,
            niter_PnP,
            has_im_poses,
            imshapes,
        )
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def compute_edge_scores(edges, pred_source, pred_target, conf_source, conf_target):
    """Compute confidence scores for each edge based on average confidence."""
    edge_to_edge_confidence_score = {}
    for edge in edges:
        _edge_str = f"{edge[0]}_{edge[1]}"
        edge_confidence_score = float(conf_source[_edge_str].mean() * conf_target[_edge_str].mean())
        edge_to_edge_confidence_score[edge] = edge_confidence_score
    return edge_to_edge_confidence_score


def construct_sparse_graph_ordered_by_negative_confidence_score(edge_to_edge_confidence_score, n_imgs):
    """Construct a sparse graph with negative confidence scores for minimum spanning tree."""
    # construct sparse graph
    sparse_graph = sp.dok_array((n_imgs, n_imgs))
    for edge, confidence_score in edge_to_edge_confidence_score.items():
        # here we use the negative of the confidence since we want to leverage
        # scipy's minimum spanning tree algorithm to construct the
        # maximum spanning tree
        sparse_graph[edge] = -confidence_score
    return sparse_graph


@dataclass
class MaximumSpanningTreeEdge:
    """Represents an edge in the maximum spanning tree with confidence score and vertices."""

    confidence: float
    source_vertex: int
    target_vertex: int

    @property
    def edge_str(self):
        return f"{self.source_vertex}_{self.target_vertex}"


def extract_maximum_spanning_tree_edges(sparse_graph_with_negative_weights):
    """
    Creates a maximum spanning tree (by using minimum spanning tree on negative weights)
    and extracts edges as MaximumSpanningTreeEdge objects.

    Args:
        sparse_graph_with_negative_weights: Sparse graph with negative confidence scores
                                            (required for maximum spanning tree)

    Returns:
        List of MaximumSpanningTreeEdge objects sorted by confidence
    """
    # Computing maximum spanning tree by using minimum spanning tree on negative weights
    maximum_spanning_tree = sp.csgraph.minimum_spanning_tree(sparse_graph_with_negative_weights).tocoo()
    edges = []

    # Convert maximum spanning tree data to MaximumSpanningTreeEdge objects
    for neg_confidence, source_vertex, target_vertex in zip(
        maximum_spanning_tree.data, maximum_spanning_tree.row, maximum_spanning_tree.col
    ):
        # Convert negative confidence back to positive
        confidence = -neg_confidence
        edges.append(MaximumSpanningTreeEdge(confidence, source_vertex, target_vertex))

    edges.sort(key=lambda edge: edge.confidence, reverse=True)

    return edges


def find_edge_to_connect_tree(remaining_edges, done):
    """Find an edge that connects to our existing tree."""
    for idx, edge in enumerate(remaining_edges):
        i, j = edge.source_vertex, edge.target_vertex

        # Check if this edge connects to our existing tree
        if (i in done and j not in done) or (j in done and i not in done):
            return idx
    return -1


def initialize_tree_with_strongest_edge(
    strongest_edge, pts3d, im_poses, im_focals, done, msp_edges, pred_source, pred_target, has_im_poses, device
):
    """Initialize the tree with the strongest edge."""
    i, j = strongest_edge.source_vertex, strongest_edge.target_vertex
    edge_str = strongest_edge.edge_str

    # Initialize the first two point clouds
    pts3d[i] = pred_source[edge_str].clone()
    pts3d[j] = pred_target[edge_str].clone()
    done.add(i)
    done.add(j)

    if has_im_poses:
        # Set the strongest edge to be the origin of the world
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_source[edge_str])

    # Add the first edge to our maximum spanning tree
    msp_edges.append((i, j))

    return pts3d, im_poses, im_focals, done, msp_edges


def process_edge_i_to_j(
    edge,
    i,
    j,
    pts3d,
    im_poses,
    done,
    msp_edges,
    pred_source,
    pred_target,
    conf_source,
    conf_target,
    device,
    has_im_poses,
):
    """Process an edge where i is in the tree and j is not."""
    edge_str = edge.edge_str
    # 1. Find transformation to align source points with existing world coordinates
    s, R, T = rigid_points_registration(pred_source[edge_str], pts3d[i], conf=conf_source[edge_str])
    trf = sRT_to_4x4(s, R, T, device)

    # 2. Transform target points to world coordinates using the same transformation
    pts3d[j] = geotrf(trf, pred_target[edge_str])

    # 3. Update tracking variables
    done.add(j)
    msp_edges.append((i, j))

    # 4. Update camera pose if needed
    if has_im_poses and im_poses[i] is None:
        im_poses[i] = sRT_to_4x4(1, R, T, device)

    return pts3d, im_poses, done, msp_edges


def process_edge_j_to_i(
    edge,
    i,
    j,
    pts3d,
    im_poses,
    done,
    msp_edges,
    pred_source,
    pred_target,
    conf_source,
    conf_target,
    device,
    has_im_poses,
):
    """Process an edge where j is in the tree and i is not."""
    edge_str = edge.edge_str
    # 1. Find transformation to align target points with existing world coordinates
    s, R, T = rigid_points_registration(pred_target[edge_str], pts3d[j], conf=conf_target[edge_str])
    trf = sRT_to_4x4(s, R, T, device)

    # 2. Transform source points to world coordinates using the same transformation
    pts3d[i] = geotrf(trf, pred_source[edge_str])

    # 3. Update tracking variables
    done.add(i)
    msp_edges.append((i, j))

    # 4. Update camera pose if needed
    if has_im_poses and im_poses[i] is None:
        im_poses[i] = sRT_to_4x4(1, R, T, device)

    return pts3d, im_poses, done, msp_edges


def grow_tree(
    remaining_edges,
    pts3d,
    im_poses,
    im_focals,
    done,
    msp_edges,
    pred_source,
    pred_target,
    conf_source,
    conf_target,
    has_im_poses,
    device,
):
    """Grow the tree by adding edges one by one."""
    while remaining_edges:
        # Find an edge that connects to the existing tree
        edge_idx = find_edge_to_connect_tree(remaining_edges, done)

        # If no connecting edge found, break (shouldn't happen in a valid spanning tree)
        if edge_idx == -1:
            break

        # Process the found edge
        edge = remaining_edges.pop(edge_idx)
        i, j = edge.source_vertex, edge.target_vertex
        edge_str = edge.edge_str

        # Estimate focal length if needed
        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_source[edge_str])

        # Case 1: i is in the tree, j is not
        if i in done and j not in done:
            pts3d, im_poses, done, msp_edges = process_edge_i_to_j(
                edge,
                i,
                j,
                pts3d,
                im_poses,
                done,
                msp_edges,
                pred_source,
                pred_target,
                conf_source,
                conf_target,
                device,
                has_im_poses,
            )

        # Case 2: j is in the tree, i is not
        elif j in done and i not in done:
            pts3d, im_poses, done, msp_edges = process_edge_j_to_i(
                edge,
                i,
                j,
                pts3d,
                im_poses,
                done,
                msp_edges,
                pred_source,
                pred_target,
                conf_source,
                conf_target,
                device,
                has_im_poses,
            )

    return pts3d, im_poses, im_focals, done, msp_edges


def complete_missing_information(
    pts3d,
    im_poses,
    im_focals,
    sparse_graph,
    pred_source,
    im_conf,
    min_conf_thr,
    device,
    niter_PnP,
    has_im_poses,
    imshapes,
):
    """
    Complete missing focal lengths and camera poses.

    This step is crucial because:
    1. Not all focal lengths may have been estimated during tree construction
    2. Some camera poses might be missing if they weren't directly connected in the MST
    3. We need complete camera parameters for all images to create a coherent 3D reconstruction

    The function works by:
    1. First estimating any missing focal lengths using the best available edges
    2. Then using Perspective-n-Point (PnP) to estimate camera poses from 3D-2D correspondences
    3. Falling back to identity poses if PnP fails (as a last resort)

    Args:
        pts3d: List of 3D points in world coordinates
        im_poses: List of camera poses (some may be None)
        im_focals: List of focal lengths (some may be None)
        sparse_graph: Graph with edge confidence scores

    Returns:
        im_poses: Complete list of camera poses
        im_focals: Complete list of focal lengths
    """
    if not has_im_poses:
        return None, None

    # First, estimate any missing focal lengths using the best available edges
    # We sort edges from best to worst based on confidence scores
    pair_scores = list(sparse_graph.values())  # already negative scores: less is best
    edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]

    # Process edges in order of confidence to estimate missing focal lengths
    for i, j in edges_from_best_to_worse.tolist():
        if im_focals[i] is None:
            edge_key = f"{i}_{j}"
            # Use the source prediction from this edge to estimate focal length
            im_focals[i] = estimate_focal(pred_source[edge_key])

    # Then, estimate any missing poses using Perspective-n-Point (PnP)
    # PnP solves for camera pose given 3D points and their 2D projections
    for i in range(len(imshapes)):
        if im_poses[i] is None:
            # Use only points with confidence above threshold for reliable pose estimation
            msk = im_conf[i] > min_conf_thr

            # fast_pnp attempts to solve the PnP problem to get camera pose
            res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)

            if res:
                # If PnP succeeds, update focal length and pose
                im_focals[i], im_poses[i] = res

        # If still no pose (PnP failed), use identity as fallback
        # This is a last resort to ensure all images have poses
        if im_poses[i] is None:
            im_poses[i] = torch.eye(4, device=device)

    # Stack all poses into a single tensor for easier handling
    im_poses = torch.stack(im_poses)

    return im_poses, im_focals


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        principal_point = torch.tensor((W / 2, H / 2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(
        pts3d_i.unsqueeze(0), principal_point.unsqueeze(0), focal_mode="weiszfeld"
    ).ravel()
    return float(focal)


def estimate_focal_knowing_depth(pts3d, principle_point, focal_mode="median", min_focal=0.0, max_focal=np.inf):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - principle_point.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
    return focal


def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True
    )
    return s, R, T  # return un-scaled (R, T)


def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and Trf.ndim == 3 and pts.ndim == 4:
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def edge_str(i, j):
    return f"{i}_{j}"


def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S / 2, S * 3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W / 2, H / 2)
    else:
        pp = to_numpy(pp)

    best = (0,)
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(
            pts3d[msk], pixels[msk], K, None, iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP
        )
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def to_numpy(x):
    return todevice(x, "numpy")


def todevice(batch, device, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x
