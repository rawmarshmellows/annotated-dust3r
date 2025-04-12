import math
from pathlib import Path

import torch
from loguru import logger

from src.annotated.croco.croco import AnnotatedCroCo
from src.annotated.losses.masked_mse import AnnotatedMaskedMSE
from src.annotated.utils.load_images import LoadConfig, load_images


def test_training():
    from icecream import ic

    ic.disable()

    annotated_model = AnnotatedCroCo(img_size=224, patch_size=16, pos_embed="RoPE100")
    loss_fn = AnnotatedMaskedMSE(norm_pix_loss=True, masked=True)
    optimizer = torch.optim.AdamW(annotated_model.parameters(), lr=0.001)

    # Create sample inputs
    image1, image2 = load_images(
        [
            str(Path(__file__).parent.parent.parent / "test_data" / "Chateau1.png"),
            str(Path(__file__).parent.parent.parent / "test_data" / "Chateau2.png"),
        ],
        config=LoadConfig(size=224),
    )
    img1 = image1.img_tensor
    img2 = image2.img_tensor

    # Forward pass
    loss_value = math.inf
    # Use tqdm for progress visualization
    from tqdm import tqdm

    pbar = tqdm(desc="Training", leave=True)

    while loss_value > 0.5:
        out, mask, target = annotated_model(img1, img2)
        loss = loss_fn(out, mask, target)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update loss value and progress bar
        loss_value = loss.item()
        pbar.set_postfix({"loss": f"{loss_value:.4f}"})
        pbar.update(1)

    pbar.close()
    logger.info(f"Final loss: {loss_value:.4f}")
