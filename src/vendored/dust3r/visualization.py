import numpy as np
import PIL
import torch
import trimesh
from scipy.spatial.transform import Rotation

from .utils import geotrf, img_to_arr, to_numpy, uint8

OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class SceneViz:
    def __init__(self):
        self.scene = trimesh.Scene()

    def add_rgbd(self, image, depth, intrinsics=None, cam2world=None, zfar=np.inf, mask=None):
        # make up some intrinsics
        if intrinsics is None:
            H, W, THREE = image.shape
            focal = max(H, W)
            intrinsics = np.float32([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

        # compute 3d points
        pts3d, mask2 = depthmap_to_absolute_camera_coordinates(depth, intrinsics, cam2world)
        mask2 &= depth < zfar

        # combine with provided mask if any
        if mask is not None:
            mask2 &= mask

        return self.add_pointcloud(pts3d, image, mask=mask2)

    def add_pointcloud(self, pts3d, color=(0, 0, 0), mask=None, denoise=False):
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if not isinstance(pts3d, list):
            pts3d = [pts3d.reshape(-1, 3)]
            if mask is not None:
                mask = [mask.ravel()]
        if not isinstance(color, (tuple, list)):
            color = [color.reshape(-1, 3)]
        if mask is None:
            mask = [slice(None)] * len(pts3d)

        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pct = trimesh.PointCloud(pts)

        if isinstance(color, (list, np.ndarray, torch.Tensor)):
            color = to_numpy(color)
            col = np.concatenate([p[m] for p, m in zip(color, mask)])
            assert col.shape == pts.shape
            pct.visual.vertex_colors = uint8(col.reshape(-1, 3))
        else:
            assert len(color) == 3
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        if denoise:
            # remove points which are noisy
            centroid = np.median(pct.vertices, axis=0)
            dist_to_centroid = np.linalg.norm(pct.vertices - centroid, axis=-1)
            dist_thr = np.quantile(dist_to_centroid, 0.99)
            valid = dist_to_centroid < dist_thr
            # new cleaned pointcloud
            pct = trimesh.PointCloud(pct.vertices[valid], color=pct.visual.vertex_colors[valid])

        self.scene.add_geometry(pct)
        return self

    def add_camera(self, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.03):
        pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
        image = img_to_arr(image)
        if isinstance(focal, np.ndarray) and focal.shape == (3, 3):
            intrinsics = focal
            focal = (intrinsics[0, 0] * intrinsics[1, 1]) ** 0.5
            if imsize is None:
                imsize = (2 * intrinsics[0, 2], 2 * intrinsics[1, 2])

        add_scene_cam(self.scene, pose_c2w, color, image, focal, imsize=imsize, screen_width=cam_size, marker=None)
        return self

    def add_cameras(self, poses, focals=None, images=None, imsizes=None, colors=None, **kw):
        get = lambda arr, idx: None if arr is None else arr[idx]
        for i, pose_c2w in enumerate(poses):
            self.add_camera(
                pose_c2w, get(focals, i), image=get(images, i), color=get(colors, i), imsize=get(imsizes, i), **kw
            )
        return self

    def show(self, point_size=2, **kw):
        self.scene.show(line_settings={"point_size": point_size}, **kw)


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam  # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = depthmap > 0.0
    return X_cam, valid_mask


def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03, marker=None):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H, W) * 1.1  # default value

    # create fake camera
    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler("z", np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95 * cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == "o":
        marker = trimesh.creation.icosphere(3, radius=screen_width / 4)
        marker.vertices += pose_c2w[:3, 3]
        marker.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(marker)
