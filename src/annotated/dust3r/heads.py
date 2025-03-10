import torch
import torch.nn as nn
import torch.nn.functional as F


def head_factory(head_type, output_mode, net, has_conf=False):
    """ " build a prediction head for the decoder"""
    if head_type == "linear" and output_mode == "pts3d":
        return LinearPts3d(net, has_conf)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_size
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf
        n_channels = 3 + has_conf
        self.proj = nn.Linear(net.decoder.embed_dim, n_channels * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)

        # feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # Manually implement pixel shuffle without using F.pixel_shuffle
        # Original shape: [B, C*r*r, H/r, W/r] -> Target shape: [B, C, H, W]
        # where r is self.patch_size
        B, C_r_squared, H_small, W_small = feat.shape
        C = C_r_squared // (self.patch_size**2)  # Calculate number of channels in output

        # Reshape to [B, C, patch_size, patch_size, H_small, W_small]
        feat = feat.view(B, C, self.patch_size, self.patch_size, H_small, W_small)

        # Permute to [B, C, H_small, patch_size, W_small, patch_size]
        feat = feat.permute(0, 1, 4, 2, 5, 3)

        # Reshape to [B, C, H_small*patch_size, W_small*patch_size] which is [B, C, H, W]
        feat = feat.reshape(B, C, H_small * self.patch_size, W_small * self.patch_size)  # B,3,H,W

        # Add debug comment
        # Now feat has shape [B, 3(+conf), H, W] where H=H_small*patch_size, W=W_small*patch_size

        # permute + norm depth
        return postprocess(feat, self.depth_mode, self.conf_mode)


def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    res = dict(pts3d=reg_dense_depth(fmap[:, :, :, 0:3], mode=depth_mode))

    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    return res


def reg_dense_depth(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds

    if mode == "linear":
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return xyz * d.square()

    if mode == "exp":
        return xyz * torch.expm1(d)

    raise ValueError(f"bad {mode=}")


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")
