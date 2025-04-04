import numpy as np
import roma
import torch
import torch.nn as nn
import tqdm
from icecream import ic

from src.annotated.dust3r.minimum_spanning_tree import minimum_spanning_tree_v2
from src.vendored.dust3r.minimum_spanning_tree import (
    align_multiple_poses,
    init_minimum_spanning_tree,
    rigid_points_registration,
    sRT_to_4x4,
)
from src.vendored.dust3r.utils import (
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
    linear_schedule,
    rgb,
    signed_expm1,
    signed_log1p,
    to_numpy,
    xy_grid,
)
from src.vendored.dust3r.visualization import SceneViz


class PointCloudOptimizer(nn.Module):
    def __init__(
        self,
        view1,
        view2,
        pred1,
        pred2,
        device,
        optimize_principle_points=False,
        focal_break=20,  # TODO: What is this parameter?
        pw_break=20,  # TODO: What is this parameter?
        dist="l2_dist",
        min_conf_thr=3,
        conf="log",
        base_scale=0.5,
    ):
        ic(self.__class__.__name__)
        ic(view1.keys())
        ic(view2.keys())
        ic(pred1.keys())
        ic(pred2.keys())
        ic(pred1["pts3d"].shape)
        ic(pred1["conf"].shape)
        ic(pred2["pts3d_in_other_view"].shape)
        ic(pred2["conf"].shape)

        super().__init__()
        self.has_im_poses = True  # by definition of this class
        self.verbose = True
        self.device = device
        self.POSE_DIM = 7

        self.focal_break = focal_break
        self.pairwise_scale = pw_break
        self.min_conf_thr = min_conf_thr
        self.confidence_transformation = get_conf_trf(conf)
        self.base_scale = base_scale
        self.dist = {
            "l1_dist": lambda a, b, weight: (a - b).norm(dim=-1) * weight,
            "l2_dist": lambda a, b, weight: (a - b).square().sum(dim=-1) * weight,
        }[dist]

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
        self.image_shapes = get_imshapes(self.edges, pred1_pts, pred2_pts)

        ic(self.image_shapes)

        # store images for later use in scene visualization
        self.imgs = None
        if "img" in view1 and "img" in view2:
            # setup images to be filled
            imgs = [torch.zeros((3, h, w)) for h, w in self.image_shapes]
            for v in range(len(self.edges)):
                idx = view1["idx"][v]
                imgs[idx] = view1["img"][v]
                idx = view2["idx"][v]
                imgs[idx] = view2["img"][v]
            self.imgs = rgb(imgs)

        # 3d point buffers
        im_areas = [h * w for h, w in self.image_shapes]
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

        # for each pixel in each image, store the maximum confidence of any edge that includes that pixel
        ic(self.edges)
        self.im_conf = nn.ParameterList([torch.zeros((h, w)) for (h, w) in self.image_shapes])
        for e, (i, j) in enumerate(self.edges):
            self.im_conf[i] = torch.maximum(self.im_conf[i], pred1_conf[e])
            self.im_conf[j] = torch.maximum(self.im_conf[j], pred2_conf[e])

        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        # confidence map buffers
        self.register_buffer(
            "_weight_i",
            ParameterStack(
                [self.confidence_transformation(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area
            ),
        )
        self.register_buffer(
            "_weight_j",
            ParameterStack(
                [self.confidence_transformation(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area
            ),
        )
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])

        # pairwise poses and scaling
        scale_dim = 1
        self.pairwise_poses = nn.Parameter(torch.randn((self.n_edges, self.POSE_DIM + scale_dim)))  # pairwise poses
        self.pairwise_adaptors = nn.Parameter(torch.zeros((self.n_edges, 2)))  # slight xy/z adaptation
        self.norm_pairwise_scale = True

        # global image poses
        self.image_poses = nn.ParameterList(torch.randn(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.image_poses = ParameterStack(self.image_poses, is_param=True)

        # depth maps
        self.image_depthmaps = nn.ParameterList(torch.randn(H, W) / 10 - 3 for H, W in self.image_shapes)  # log(depth)
        self.image_depthmaps = ParameterStack(self.image_depthmaps, is_param=True, fill=self.max_area)

        # focal
        self.image_focals = nn.ParameterList(
            torch.FloatTensor([self.focal_break * np.log(max(H, W))]) for H, W in self.image_shapes
        )  # camera intrinsics
        self.image_focals = ParameterStack(self.image_focals, is_param=True)

        # principle points
        self.image_principle_points = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))
        self.image_principle_points.requires_grad_(optimize_principle_points)

        # camera intrinsics
        self.image_principle_points = ParameterStack(self.image_principle_points, is_param=True)
        self.register_buffer("_pp", torch.tensor([(w / 2, h / 2) for h, w in self.image_shapes]))
        self.register_buffer(
            "_grid",
            ParameterStack([xy_grid(W, H, device=self.device) for H, W in self.image_shapes], fill=self.max_area),
        )

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.image_shapes]

    @property
    def str_edges(self):
        return [f"{i}_{j}" for i, j in self.edges]

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
        return self.global_alignment_loop(**kw)

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment_v2(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            device = self.device

            pts3d, _, im_focals, im_poses = minimum_spanning_tree_v2(
                self.image_shapes,
                self.edges,
                self.pred_i,
                self.pred_j,
                self.conf_i,
                self.conf_j,
                self.im_conf,
                self.min_conf_thr,
                device,
                has_im_poses=self.has_im_poses,
            )

            self.init_from_pts3d_v2(pts3d, im_focals, im_poses)

        return self.global_alignment_loop(**kw)

    def init_from_pts3d_v2(self, pts3d, im_focals, im_poses):
        # init poses
        nkp, known_poses_msk, known_poses = self.get_known_poses()

        import pdb

        pdb.set_trace()

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
            self._set_pose(self.pairwise_poses, e, R, T, scale=s)

        # take into account the scale normalization
        s_factor = self.pw_norm_scale_factor
        im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
        for img_pts3d in pts3d:
            img_pts3d *= s_factor

        # init all image poses
        if self.has_im_poses:
            for i in range(self.n_imgs):
                cam2world = im_poses[i]
                depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
                self._set_depthmap(i, depth)
                self._set_pose(self.image_poses, i, cam2world)
                if im_focals[i] is not None:
                    self._set_focal(i, im_focals[i])

        if self.verbose:
            print(" init loss =", float(self()))

    def global_alignment_loop(self, lr=0.01, niter=300, schedule="cosine", lr_min=1e-6):
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            return self

        if self.verbose:
            print("Global alignement - optimizing for:")
            print([name for name, value in self.named_parameters() if value.requires_grad])

        lr_base = lr
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

        loss = float("inf")
        if self.verbose:
            with tqdm.tqdm(total=niter) as bar:
                while bar.n < bar.total:
                    loss, lr = self.global_alignment_iter(bar.n, niter, lr_base, lr_min, optimizer, schedule)
                    bar.set_postfix_str(f"{lr=:g} loss={loss:g}")
                    bar.update()
        else:
            for n in range(niter):
                loss, _ = self.global_alignment_iter(n, niter, lr_base, lr_min, optimizer, schedule)
        return loss

    def global_alignment_iter(self, cur_iter, niter, lr_base, lr_min, optimizer, schedule):
        t = cur_iter / niter
        if schedule == "cosine":
            lr = cosine_schedule(t, lr_base, lr_min)
        elif schedule == "linear":
            lr = linear_schedule(t, lr_base, lr_min)
        else:
            raise ValueError(f"bad lr {schedule=}")
        adjust_learning_rate_by_lr(optimizer, lr)
        optimizer.zero_grad()
        loss = self()
        loss.backward()
        optimizer.step()

        return float(loss), lr

    def get_known_poses(self):
        if self.has_im_poses:
            known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.image_poses])
            known_poses = self.im_poses
            return known_poses_msk.sum(), known_poses_msk, known_poses
        else:
            return 0, None, None

    def forward(self):
        pw_poses = self.pw_poses  # cam-to-world
        pw_adapt = self.adaptors.unsqueeze(1)
        proj_pts3d = self.pts3d(raw=True)  # these are initially all randomly initialized

        # Rotate pairwise prediction according to scaled_pw_poses
        # _stacked_pred_i, and _stacked_pred_j are the depth maps generated from Dust3r
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # Compute the loss, the idea here is that by jointly optimizing the:
        # 1. depthmap
        # 2. principle point
        # 3. camera poses
        # we will be able to estimate them the a joint manner
        # the forward function is called only after we used the compute_global_alignment
        # which uses the depthmap generated from the to create the initial starting values for
        # the parameters that we are estimating
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        return li + lj

    @property
    def pw_poses(self):  # cam to world
        RT = to_rigid_unit_quaternion(self.pairwise_poses)
        scaled_RT = RT.clone()
        # scaled_RT[:, :3] is the first 3 rows of the quaternion
        scaled_RT[:, :3] *= self.pw_scale.view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT

    @property
    def pw_scale(self):
        scale = self.pairwise_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.pw_norm_scale_factor
        return scale

    @property
    def pw_norm_scale_factor(self):
        if self.norm_pairwise_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pairwise_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    @property
    def im_poses(self):  # cam to world
        cam2world = to_rigid_unit_quaternion(self.image_poses)
        return cam2world

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

    @property
    def adaptors(self):
        """
        This function retrieves and processes the pairwise adaptors used for scaling 3D points.

        The function:
        1. Gets the pairwise adaptors (scale factors for xy and z dimensions)
        2. Converts the 2D adaptors (xy, z) into 3D adaptors (x, y, z) by duplicating the xy value
        3. If normalization is enabled, centers the adaptors to ensure their product equals 1
        4. Applies exponential scaling after dividing by pw_break to control the magnitude

        Returns:
            torch.Tensor: Exponentially scaled adaptation factors for 3D points
        """
        # Get the raw pairwise adaptors
        adapt = self.pairwise_adaptors  # Shape: (n_edges, 2) with [scale_xy, scale_z]

        # Extract individual components for clarity
        scale_xy = adapt[:, 0:1]  # First column represents xy scale
        scale_z = adapt[:, 1:2]  # Second column represents z scale

        # Create 3D scaling vector by duplicating xy scale for both x and y dimensions
        adapt_3d = torch.cat((scale_xy, scale_xy, scale_z), dim=-1)  # Shape: (n_edges, 3)

        # Normalize adaptors if enabled (ensures product of scales ≈ 1)
        if self.norm_pairwise_scale:
            # Subtract mean to center values, making their product approximately 1
            adapt_3d = adapt_3d - adapt_3d.mean(dim=1, keepdim=True)

        # Apply exponential scaling after dividing by pairwise_scale to control magnitude
        return (adapt_3d / self.pairwise_scale).exp()

    def pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[: h * w].view(h, w, 3) for dm, (h, w) in zip(res, self.image_shapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and projection params if not provided
        focals = self.focals
        pp = self.principal_points
        im_poses = self.im_poses
        depth = self.depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    @property
    def focals(self):
        log_focals = torch.stack(list(self.image_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def _set_focal(self, idx, focal, force=False):
        param = self.image_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    @property
    def principal_points(self):
        return self._pp + 10 * self.image_principle_points

    def depthmaps(self, raw=False):
        res = self.image_depthmaps.exp()
        if not raw:
            res = [dm[: h * w].view(h, w) for dm, (h, w) in zip(res, self.image_shapes)]
        return res

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.image_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.pts3d()[n], colors[n], self.masks[n])
        else:
            viz.add_pointcloud(self.pts3d(), self.imgs, self.masks)
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.im_poses)
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(
            im_poses, self.focals, colors=colors, images=self.imgs, imsizes=self.imsizes, cam_size=cam_size
        )
        if show_pw_cams:
            pw_poses = self.pw_poses
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz

    @property
    def masks(self):
        return [(conf > self.min_conf_thr) for conf in self.im_conf]

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), "bad pair indices: missing values "
        return len(indices)


def to_rigid_unit_quaternion(poses):
    # normalize rotation
    Q = poses[:, :4]
    T = signed_expm1(poses[:, 4:7])
    RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
    return RT


def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)
