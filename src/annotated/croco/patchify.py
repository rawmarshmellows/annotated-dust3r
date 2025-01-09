def patchify(imgs, patch_size):
    """
    Divide images into non-overlapping square patches.

    Parameters:
    - imgs (torch.Tensor): Input images of shape (B, C, H, W).
    - patch_size (int): Size of each square patch (p).

    Returns:
    - patches (torch.Tensor): Patches of shape (B, L, p^2 * C),
                              where L = (H // p) * (W // p).
    - num_patches_h (int): Number of patches along the height.
    - num_patches_w (int): Number of patches along the width.
    """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions must be divisible by the patch size."

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Reshape to (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = imgs.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)

    # Permute to (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
    x = x.permute(0, 2, 4, 3, 5, 1)

    # Reshape to (B, L, p^2 * C), where L = num_patches_h * num_patches_w
    patches = x.reshape(B, num_patches_h * num_patches_w, patch_size * patch_size * C)

    return patches, num_patches_h, num_patches_w


def unpatchify(patches, patch_size, num_patches_h, num_patches_w, channels=3):
    """
    Reconstruct images from patches.

    Parameters:
    - patches (torch.Tensor): Patches of shape (B, L, p^2 * C),
                              where L = (H // p) * (W // p).
    - patch_size (int): Size of each square patch (p).
    - num_patches_h (int): Number of patches along the height.
    - num_patches_w (int): Number of patches along the width.
    - channels (int): Number of channels in the image (default: 3).

    Returns:
    - imgs (torch.Tensor): Reconstructed images of shape (B, C, H, W).
    """
    B, L, patch_dim = patches.shape
    assert patch_dim % channels == 0, "Patch dimension is not compatible with the number of channels."

    p_squared = patch_dim // channels
    p = int(patch_size)
    assert p * p == p_squared, "Patch size does not match patch dimension."

    expected_L = num_patches_h * num_patches_w
    assert L == expected_L, f"Number of patches (L={L}) does not match num_patches_h * num_patches_w ({expected_L})."

    # Reshape to (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
    x = patches.reshape(B, num_patches_h, num_patches_w, patch_size, patch_size, channels)

    # Permute to (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    x = x.permute(0, 5, 1, 3, 2, 4)

    # Reshape to (B, C, H, W), where H = num_patches_h * patch_size, W = num_patches_w * patch_size
    H = num_patches_h * patch_size
    W = num_patches_w * patch_size
    imgs = x.reshape(B, channels, H, W)

    return imgs
