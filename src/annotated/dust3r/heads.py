# TODO: Refactor

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from icecream import ic

from .dpt_block import DPTOutputAdapter  # noqa


def head_factory(head_type, output_mode, net, has_conf=False):
    """ " build a prediction head for the decoder"""
    if head_type == "linear" and output_mode == "pts3d":
        return LinearPts3d(net, has_conf)
    elif head_type == "dpt" and output_mode == "pts3d":
        return create_dpt_head(net, has_conf=has_conf)
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
        self.proj = nn.Linear(net.decoder.embed_dim, n_channels * self.patch_size[0] * self.patch_size[1])

        ic("LinearPts3d.__init__")
        ic(self.patch_size)
        ic(self.depth_mode)
        ic(self.conf_mode)
        ic(self.has_conf)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        ic("LinearPts3d.forward")
        ic(self.patch_size)
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size[0], W // self.patch_size[1])

        # feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # Manually implement pixel shuffle without using F.pixel_shuffle
        # Original shape: [B, C*r*r, H/r, W/r] -> Target shape: [B, C, H, W]
        # where r is self.patch_size
        B, C_r_squared, H_small, W_small = feat.shape
        C = C_r_squared // (self.patch_size[0] * self.patch_size[1])  # Calculate number of channels in output

        # Reshape to [B, C, patch_size, patch_size, H_small, W_small]
        feat = feat.view(B, C, self.patch_size[0], self.patch_size[1], H_small, W_small)

        # Permute to [B, C, H_small, patch_size, W_small, patch_size]
        feat = feat.permute(0, 1, 4, 2, 5, 3)

        # Reshape to [B, C, H_small*patch_size, W_small*patch_size] which is [B, C, H, W]
        feat = feat.reshape(B, C, H_small * self.patch_size[0], W_small * self.patch_size[1])  # B,3,H,W

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


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) function first"
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        n_cls_token=0,
        hooks_idx=None,
        dim_tokens=None,
        output_width_ratio=1,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs,
    ):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim // 2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan + has_conf,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=net.depth_mode,
        conf_mode=net.conf_mode,
        head_type="regression",
    )
