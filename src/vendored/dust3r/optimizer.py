import numpy as np
import roma
import torch
import torch.nn as nn
import tqdm

from .minimum_spanning_tree import (
    align_multiple_poses,
    minimum_spanning_tree_v2,
    rigid_points_registration,
    sRT_to_4x4,
)
from .utils import (
    ParameterStack,
    _ravel_hw,
    adjust_learning_rate_by_lr,
    auto_cam_size,
    cosine_schedule,
    edge_str,
    geotrf,
    get_conf_trf,
    get_imshapes,
    inv,
    l1_dist,
    l2_dist,
    linear_schedule,
    rgb,
    signed_expm1,
    signed_log1p,
    to_numpy,
    xy_grid,
)
from .visualization import SceneViz


class PointCloudOptimizer(nn.Module):
    def __init__(
        self,
        view1,
        view2,
        pred1,
        pred2,
        device,
        optimize_pp=False,
        focal_break=20,  # TODO: What is this parameter?
        pw_break=20,  # TODO: What is this parameter?
        dist="l2_dist",
        min_conf_thr=3,
        conf="log",
        base_scale=0.5,
    ):
        super().__init__()
        self.has_im_poses = True  # by definition of this class
        self.verbose = True
        self.device = device
        self.POSE_DIM = 7

        self.focal_break = focal_break
        self.pw_break = pw_break
        self.rand_pose = torch.randn
        self.min_conf_thr = min_conf_thr
        self.conf_trf = get_conf_trf(conf)
        self.base_scale = base_scale
        self.dist = {"l1_dist": l1_dist, "l2_dist": l2_dist}[dist]

        # edges in view
        self.edges = [(int(i), int(j)) for i, j in zip(view1["idx"], view2["idx"])]  # edges in view
        self.n_edges = len(self.edges)
        self.n_imgs = self._check_edges()
        self.register_buffer("_ei", torch.tensor([i for i, j in self.edges]))
        self.register_buffer("_ej", torch.tensor([j for i, j in self.edges]))

        # predictions from model
        pred1_pts = pred1["pts3d"]
        pred2_pts = pred2["pts3d_in_other_view"]
        pred1_conf = pred1["conf"]
        pred2_conf = pred2["conf"]
        self.imshapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        # store images for later use in scene visualization
        self.imgs = None
        if "img" in view1 and "img" in view2:
            imgs = [torch.zeros((3,) + hw) for hw in self.imshapes]
            for v in range(len(self.edges)):
                idx = view1["idx"][v]
                imgs[idx] = view1["img"][v]
                idx = view2["idx"][v]
                imgs[idx] = view2["img"][v]
            self.imgs = rgb(imgs)

        # 3d point buffers
        im_areas = [h * w for h, w in self.imshapes]
        self.max_area = max(im_areas)
        self.pred_i = nn.ParameterDict({ij: pred1_pts[n] for n, ij in enumerate(self.str_edges)}).requires_grad_(False)
        self.pred_j = nn.ParameterDict({ij: pred2_pts[n] for n, ij in enumerate(self.str_edges)}).requires_grad_(False)
        self.register_buffer("_stacked_pred_i", ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer("_stacked_pred_j", ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.conf_i = nn.ParameterDict({ij: pred1_conf[n] for n, ij in enumerate(self.str_edges)}).requires_grad_(
            False
        )
        self.conf_j = nn.ParameterDict({ij: pred2_conf[n] for n, ij in enumerate(self.str_edges)}).requires_grad_(
            False
        )

        self.im_conf = nn.ParameterList([torch.zeros((h, w)) for (h, w) in self.imshapes])

        for e, (i, j) in enumerate(self.edges):
            self.im_conf[i] = torch.maximum(self.im_conf[i], pred1_conf[e])
            self.im_conf[j] = torch.maximum(self.im_conf[j], pred2_conf[e])

        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        # confidence map buffers
        self.register_buffer(
            "_weight_i",
            ParameterStack([self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area),
        )
        self.register_buffer(
            "_weight_j",
            ParameterStack([self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area),
        )
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        # pairwise poses and scaling
        self.pw_poses = nn.Parameter(self.rand_pose((self.n_edges, 1 + self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.norm_pw_scale = True

        # global image poses
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.im_poses = ParameterStack(self.im_poses, is_param=True)

        # depth maps
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W) / 10 - 3 for H, W in self.imshapes)  # log(depth)
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)

        # focal
        self.im_focals = nn.ParameterList(
            torch.FloatTensor([self.focal_break * np.log(max(H, W))]) for H, W in self.imshapes
        )  # camera intrinsics
        self.im_focals = ParameterStack(self.im_focals, is_param=True)

        # principle points
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer("_pp", torch.tensor([(w / 2, h / 2) for h, w in self.imshapes]))
        self.register_buffer(
            "_grid", ParameterStack([xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area)
        )

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def str_edges(self):
        return [f"{i}_{j}" for i, j in self.edges]

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
        return global_alignment_loop(self, **kw)

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment_v2(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            device = self.device

            pts3d, _, im_focals, im_poses = minimum_spanning_tree_v2(
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
            )

            self.init_from_pts3d_v2(pts3d, im_focals, im_poses)

        return global_alignment_loop(self, **kw)

    def init_from_pts3d_v2(self, pts3d, im_focals, im_poses):
        # init poses
        nkp, known_poses_msk, known_poses = self.get_known_poses()
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

    def get_known_poses(self):
        if self.has_im_poses:
            known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
            known_poses = self.get_im_poses()
            return known_poses_msk.sum(), known_poses_msk, known_poses
        else:
            return 0, None, None

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        return li + lj

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _get_poses(self, poses):
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[: h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[: h * w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
        else:
            viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.get_im_poses())
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(
            im_poses, self.get_focals(), colors=colors, images=self.imgs, imsizes=self.imsizes, cam_size=cam_size
        )
        if show_pw_cams:
            pw_poses = self.get_pw_poses()
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz

    def get_masks(self):
        return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), "bad pair indices: missing values "
        return len(indices)


def global_alignment_loop(net, lr=0.01, niter=300, schedule="cosine", lr_min=1e-6):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print("Global alignement - optimizing for:")
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float("inf")
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule)
                bar.set_postfix_str(f"{lr=:g} loss={loss:g}")
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    t = cur_iter / niter
    if schedule == "cosine":
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == "linear":
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f"bad lr {schedule=}")
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()
    loss = net()
    loss.backward()
    optimizer.step()

    return float(loss), lr


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
