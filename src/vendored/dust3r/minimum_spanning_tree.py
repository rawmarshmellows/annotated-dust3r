from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import roma
import scipy.sparse as sp
import torch

from .utils import compute_edge_scores, edge_str, geotrf, get_med_dist_between_poses, i_j_ij, inv, to_numpy, xy_grid


def minimum_spanning_tree_v2(
    imshapes,
    edges,
    pred_i,
    pred_j,
    conf_i,
    conf_j,
    im_conf,
    min_conf_thr,
    device,
    has_im_poses=True,
    niter_PnP=10,
    verbose=True,
):
    """
    1. Calculate the mean edge score, computed from the average of each pixel in image
    2. Construct the minimum spanning tree of the negated edge scores (which gives us global scores)
    3. Construct the global coordinate point map
    4. Calculate the focals for each image
    5. Calculate the global poses for each image using PnP
    """
    n_imgs = len(imshapes)

    # calculate edge scores
    edge_to_edge_confidence_score = {}
    for edge in edges:
        _edge_str = f"{edge[0]}_{edge[1]}"
        edge_confidence_score = float(conf_i[_edge_str].mean() * conf_j[_edge_str].mean())
        edge_to_edge_confidence_score[edge] = edge_confidence_score

    # construct sparse graph
    sparse_graph = sp.dok_array((n_imgs, n_imgs))
    for edge, confidence_score in edge_to_edge_confidence_score.items():
        # here we use the negative of the confidence since we want to leverage
        # scipy's minimum spanning tree algorithm to construct the
        # maximum spanning tree
        sparse_graph[edge] = -confidence_score

    # create msp
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # create a list of tuples for (confidence_score, vertex_0_for_edge, vertex_1_for_edge)
    todo = []
    for negative_confidence_score, vertex_0_for_edge, vertex_1_for_edge in zip(msp.data, msp.row, msp.col):
        # here to negate the confidence to get back the original value
        todo.append((-negative_confidence_score, vertex_0_for_edge, vertex_1_for_edge))

    # temp variable to store 3d points
    # pts3d is the pointmap in world coordinates
    pts3d = [None] * len(imshapes)

    # im_poses is the pose in world coordinates
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f" init edge ({i}*,{j}*) {score=}")
    i_j = f"{i}_{j}"
    pts3d[i] = pred_i[i_j].clone()
    pts3d[j] = pred_j[i_j].clone()
    done = {i, j}
    if has_im_poses:
        # set the strongest edge to be the origin of the world
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]

    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if verbose:
            print(f" init edge ({i},{j}*) {score=}")

        i_j = f"{i}_{j}"

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_i[i_j])

        if i in done:
            assert j not in done
            # pred_i[i_j] is the depthmap for image i in the coordinate frame of i
            # pred_j[i_j] is the depthmap for image j in the coordinate frame of i
            # pts3d[i] is the depthmap for image in in the world coordinate frame

            # 1. Find the the scale, rotation, and translation to align pred_i[i_j] with pts3d[i]
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])

            # 2. Convert to homogeneous coordinates
            trf = sRT_to_4x4(s, R, T, device)

            # 3. Use the same matrix to transform pointmap pred_j[i_j] to world coordinates
            #    this is possible as pred_j[i_j] is in the same coordinate frame as pred_i[i_j]
            pts3d[j] = geotrf(trf, pred_j[i_j])

            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:
            assert i not in done
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(sparse_graph.values())  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])

        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def init_from_pts3d(self, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)

    # set all pairwise poses
    for e, (i, j) in enumerate(self.edges):
        i_j = f"{i}_{j}"
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                self._set_focal(i, im_focals[i])

    if self.verbose:
        print(" init loss =", float(self()))


@torch.no_grad()
def init_minimum_spanning_tree(self, **kw):
    """Init all camera poses (image-wise and pairwise poses) given
    an initial set of pairwise estimations.
    """
    device = self.device
    pts3d, _, im_focals, im_poses = minimum_spanning_tree(
        self.imshapes,
        self.edges,
        self.pred_i,
        self.pred_j,
        self.conf_i,
        self.conf_j,
        self.im_conf,
        self.min_conf_thr,
        device,
        has_im_poses=self.has_im_poses,
        verbose=self.verbose,
        **kw,
    )

    return init_from_pts3d(self, pts3d, im_focals, im_poses)


def minimum_spanning_tree(
    imshapes,
    edges,
    pred_i,
    pred_j,
    conf_i,
    conf_j,
    im_conf,
    min_conf_thr,
    device,
    has_im_poses=True,
    niter_PnP=10,
    verbose=True,
):
    # import pickle

    # function_inputs = {
    #     "imshapes": imshapes,
    #     "edges": edges,
    #     "pred_i": pred_i,
    #     "pred_j": pred_j,
    #     "conf_i": conf_i,
    #     "conf_j": conf_j,
    #     "im_conf": im_conf,
    #     "min_conf_thr": min_conf_thr,
    #     "device": device,
    #     "has_im_poses": has_im_poses,
    #     "niter_PnP": niter_PnP,
    #     "verbose": verbose,
    # }
    # pickle.dump(function_inputs, open("msp_inputs.p", "wb"))

    n_imgs = len(imshapes)
    sparse_graph = -dict_to_sparse_graph(compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j))
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f" init edge ({i}*,{j}*) {score=}")
    i_j = edge_str(i, j)
    pts3d[i] = pred_i[i_j].clone()
    pts3d[j] = pred_j[i_j].clone()
    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]
    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_i[i_j])

        if i in done:
            if verbose:
                print(f" init edge ({i},{j}*) {score=}")
            assert j not in done
            # align pred[i] with pts3d[i], and then set j accordingly
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[j] = geotrf(trf, pred_j[i_j])
            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:
            if verbose:
                print(f" init edge ({i}*,{j}) {score=}")
            assert i not in done
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(sparse_graph.values())  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])

        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def dict_to_sparse_graph(dic):
    n_imgs = max(max(e) for e in dic) + 1
    res = sp.dok_array((n_imgs, n_imgs))
    for edge, value in dic.items():
        res[edge] = value
    return res


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


def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


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


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode="weiszfeld").ravel()
    return float(focal)


def estimate_focal_knowing_depth(pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
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


def get_known_poses(self):
    if self.has_im_poses:
        known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
        known_poses = self.get_im_poses()
        return known_poses_msk.sum(), known_poses_msk, known_poses
    else:
        return 0, None, None


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps * poses[:, :3, 2]))

    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    return s, R, T
