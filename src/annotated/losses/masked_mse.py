import torch.nn as nn


class MaskedMSE(nn.Module):
    def __init__(self, norm_pix_loss=False, masked=True):
        """
        norm_pix_loss: normalize each patch by their pixel mean and variance
        masked: compute loss over the masked patches only
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked

    def forward(self, pred, mask, target):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.masked:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches
        else:
            loss = loss.mean()  # mean loss
        return loss


class AnnotatedMaskedMSE(nn.Module):
    """
    Computes masked mean squared error loss between predictions and targets.

    Args:
        norm_pix_loss (bool): Whether to normalize each patch by pixel mean and variance
        masked (bool): Whether to compute loss only over masked patches

    Debug:
        - Forward pass normalizes target if norm_pix_loss=True
        - Computes MSE loss per patch
        - If masked=True, averages loss only over masked patches
        - If masked=False, averages loss over all patches
    """

    def __init__(self, norm_pix_loss=False, masked=True):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked

    def forward(self, pred, mask, target):
        # Debug: Normalize target patches if enabled
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # Debug: Compute MSE loss per patch
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Shape: [N, L], mean loss per patch

        # Debug: Average loss over masked or all patches
        if self.masked:
            loss = (loss * mask).sum() / mask.sum()  # Average over masked patches
        else:
            loss = loss.mean()  # Average over all patches

        return loss
