# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

from .postprocess import postprocess


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf) * self.patch_size**2)

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
