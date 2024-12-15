import pytest
import torch
import torch.nn as nn

from src.annotated.vision_transformer import VisionTransformerEncoder as AnnotatedEncoder
from src.croco.models.croco import CroCoNet


def test_vision_transformer_encoder_equivalence():
    """Test equivalence between Annotated and CroCo Encoder implementations"""

    # Test parameters
    batch_size = 2
    img_size = 224
    embed_dim = 768
    depth = 12
    num_heads = 12
    mlp_ratio = 4.0

    # Create random input with fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize both implementations with same parameters
    annotated_encoder = AnnotatedEncoder(
        img_size=img_size,
        patch_size=16,
        embed_dim=embed_dim,
        num_layers=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        embed_norm_layer=nn.Identity,  # Match CroCo's lack of normalization in patch embed
        pos_embed_type="sincos2d",
    )

    croco = CroCoNet(
        img_size=img_size,
        patch_size=16,
        enc_embed_dim=embed_dim,
        enc_depth=depth,
        enc_num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        pos_embed="cosine",  # Ensure we're using the same positional embedding type
    )

    # Set both to eval mode and move to CPU to ensure deterministic behavior
    annotated_encoder.eval()
    croco.eval()

    # Initialize weights identically
    with torch.no_grad():
        # Map the state dict keys from croco_encoder to annotated_encoder
        croco_state_dict = croco.state_dict()

        # Create mapping between CroCo and Annotated keys
        key_mapping = {
            # Positional embedding and patch embedding
            "enc_pos_embed": "enc_pos_embed",
            # Note here that CroCo's norm_layer for the PatchEmbed is nn.Identity, so we don't need to map it
            # I believe this is a bug with their code
            # "patch_embed.norm.weight": "patch_embed.norm.weight",
            # "patch_embed.norm.bias": "patch_embed.norm.bias",
            "patch_embed.proj.weight": "patch_embed.proj.weight",
            "patch_embed.proj.bias": "patch_embed.proj.bias",
            "enc_norm.weight": "norm.weight",
            "enc_norm.bias": "norm.bias",
        }

        # Add mappings for each encoder block
        for i in range(depth):
            # Attention layers
            key_mapping[f"enc_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.attn.query_key_value.weight"
            key_mapping[f"enc_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.attn.query_key_value.bias"
            key_mapping[f"enc_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.proj.weight"
            key_mapping[f"enc_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.proj.bias"

            # Norm layers
            key_mapping[f"enc_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
            key_mapping[f"enc_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
            key_mapping[f"enc_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
            key_mapping[f"enc_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"

            # MLP layers
            key_mapping[f"enc_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
            key_mapping[f"enc_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
            key_mapping[f"enc_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
            key_mapping[f"enc_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"

        # Create new state dict with mapped keys
        new_state_dict = {}
        missing_keys_in_croco = []
        for croco_key, annotated_key in key_mapping.items():
            if croco_key in croco_state_dict:
                new_state_dict[annotated_key] = croco_state_dict[croco_key]
            else:
                missing_keys_in_croco.append(croco_key)

        # make sure all keys in annotated encoder are present in the new state dict
        for key in annotated_encoder.state_dict().keys():
            if key not in new_state_dict:
                raise ValueError(f"Key {key} not found in new state dict")

        # Load the mapped state dict into annotated encoder
        missing_keys, unexpected_keys = annotated_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"\nMissing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        print(f"Missing keys in croco: {missing_keys_in_croco}")

        if any(missing_keys) or any(unexpected_keys) or any(missing_keys_in_croco):
            raise ValueError("State dict mapping failed")

    # Set same random seed for masking
    torch.manual_seed(42)

    # Forward pass
    with torch.no_grad():
        # First get patch embeddings from both to compare
        torch.manual_seed(42)
        x_croco, pos_croco = croco.patch_embed(x)
        torch.manual_seed(42)
        x_annotated, pos_annotated = annotated_encoder.patch_embed(x)

        print("\nPatch Embedding Comparison:")
        print(f"Shapes match: {x_croco.shape == x_annotated.shape}")
        assert x_croco.shape == x_annotated.shape, "Patch embedding shapes do not match"

        patch_max_diff = torch.max(torch.abs(x_croco - x_annotated)).item()
        print(f"Max difference in patch embeddings: {patch_max_diff:.2e}")
        assert patch_max_diff < 1e-6, "Patch embeddings differ significantly"

        # Now get the full forward pass
        torch.manual_seed(42)
        croco_feat, croco_pos, croco_mask = croco._encode_image(x, do_mask=True)
        torch.manual_seed(42)
        annotated_feat, annotated_pos, annotated_mask = annotated_encoder(x, do_mask=True)

    # Compare outputs
    max_diff = torch.max(torch.abs(annotated_feat - croco_feat)).item()
    mean_diff = torch.mean(torch.abs(annotated_feat - croco_feat)).item()

    print("\nMask Comparison:")
    print(f"Mask shapes match: {croco_mask.shape == annotated_mask.shape}")
    assert croco_mask.shape == annotated_mask.shape, "Mask shapes do not match"

    if croco_mask.shape == annotated_mask.shape:
        mask_diff = torch.sum(croco_mask != annotated_mask).item()
        print(f"Number of differing mask positions: {mask_diff}")
        assert mask_diff == 0, "Masks differ in content"

    print("\nEncoder Implementation Comparison:")
    print(f"Max difference: {max_diff:.2e}")
    assert max_diff < 1e-6, "Maximum difference exceeds tolerance"

    print(f"Mean difference: {mean_diff:.2e}")
    assert mean_diff < 1e-6, "Mean difference exceeds tolerance"

    print(f"Shapes match: {annotated_feat.shape == croco_feat.shape}")
    assert annotated_feat.shape == croco_feat.shape, "Output shapes do not match"

    print(f"Output shape: {annotated_feat.shape}")

    # Test with different input patterns
    test_cases = {
        "Uniform": torch.ones_like(x),
        "Random": torch.randn_like(x),
        "Zero": torch.zeros_like(x),
        "Large Values": torch.ones_like(x) * 100,
    }

    for name, test_input in test_cases.items():
        with torch.no_grad():
            torch.manual_seed(42)  # Reset seed for each test case
            croco_feat, croco_pos, croco_mask = croco._encode_image(test_input, do_mask=True)

            torch.manual_seed(42)  # Reset seed for each test case
            annotated_feat, annotated_pos, annotated_mask = annotated_encoder(test_input, do_mask=True)

            max_diff = torch.max(torch.abs(annotated_feat - croco_feat)).item()
            print(f"\n{name} Input Test:")
            print(f"Max difference: {max_diff:.2e}")
            assert max_diff < 1e-6, f"Maximum difference exceeds tolerance for {name} input"

            print(f"Outputs match: {torch.allclose(croco_feat, annotated_feat, atol=1e-6)}")
            assert torch.allclose(croco_feat, annotated_feat, atol=1e-6), f"Outputs do not match for {name} input"
