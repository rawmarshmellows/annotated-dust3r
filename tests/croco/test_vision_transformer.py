import numpy as np
import pytest
import torch
import torch.nn as nn

from src.annotated.vision_transformer import VisionTransformerDecoder as AnnotatedDecoder
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


def test_vision_transformer_decoder_equivalence():
    """Test equivalence between Annotated and CroCo Decoder implementations"""

    # Test parameters
    batch_size = 2
    img_size = 224
    enc_embed_dim = 768
    dec_embed_dim = 512
    enc_depth = 12
    dec_depth = 8
    enc_num_heads = 12
    dec_num_heads = 8
    mlp_ratio = 4.0

    # Create random input with fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(batch_size, 3, img_size, img_size)

    # Initialize both implementations with same parameters
    annotated_decoder = AnnotatedDecoder(
        img_size=img_size,
        patch_size=16,
        enc_embed_dim=enc_embed_dim,
        embed_dim=dec_embed_dim,
        num_layers=dec_depth,
        num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        path_drop_rate=0.0,
        pos_embed_type="sincos2d",
    )

    croco = CroCoNet(
        img_size=img_size,
        patch_size=16,
        enc_embed_dim=enc_embed_dim,
        dec_embed_dim=dec_embed_dim,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        enc_num_heads=enc_num_heads,
        dec_num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        pos_embed="cosine",
    )

    # Set both to eval mode
    annotated_decoder.eval()
    croco.eval()

    # Initialize weights identically
    with torch.no_grad():
        croco_state_dict = croco.state_dict()

        # Create mapping between CroCo and Annotated decoder keys
        key_mapping = {
            "dec_pos_embed": "dec_pos_embed",
            "decoder_embed.weight": "decoder_embed.weight",
            "decoder_embed.bias": "decoder_embed.bias",
            "dec_norm.weight": "norm.weight",
            "dec_norm.bias": "norm.bias",
            "mask_token": "mask_token",
        }

        # Add mappings for each decoder block
        for i in range(dec_depth):
            # Attention layers
            key_mapping[f"dec_blocks.{i}.attn.qkv.weight"] = f"blocks.{i}.attn.query_key_value.weight"
            key_mapping[f"dec_blocks.{i}.attn.qkv.bias"] = f"blocks.{i}.attn.query_key_value.bias"
            key_mapping[f"dec_blocks.{i}.attn.proj.weight"] = f"blocks.{i}.attn.proj.weight"
            key_mapping[f"dec_blocks.{i}.attn.proj.bias"] = f"blocks.{i}.attn.proj.bias"

            # Cross attention layers
            key_mapping[f"dec_blocks.{i}.cross_attn.projq.weight"] = f"blocks.{i}.cross_attn.query_projection.weight"
            key_mapping[f"dec_blocks.{i}.cross_attn.projq.bias"] = f"blocks.{i}.cross_attn.query_projection.bias"
            key_mapping[f"dec_blocks.{i}.cross_attn.projk.weight"] = f"blocks.{i}.cross_attn.key_projection.weight"
            key_mapping[f"dec_blocks.{i}.cross_attn.projk.bias"] = f"blocks.{i}.cross_attn.key_projection.bias"
            key_mapping[f"dec_blocks.{i}.cross_attn.projv.weight"] = f"blocks.{i}.cross_attn.value_projection.weight"
            key_mapping[f"dec_blocks.{i}.cross_attn.projv.bias"] = f"blocks.{i}.cross_attn.value_projection.bias"
            key_mapping[f"dec_blocks.{i}.cross_attn.proj.weight"] = f"blocks.{i}.cross_attn.output_projection.weight"
            key_mapping[f"dec_blocks.{i}.cross_attn.proj.bias"] = f"blocks.{i}.cross_attn.output_projection.bias"

            # Norm layers
            key_mapping[f"dec_blocks.{i}.norm1.weight"] = f"blocks.{i}.norm1.weight"
            key_mapping[f"dec_blocks.{i}.norm1.bias"] = f"blocks.{i}.norm1.bias"
            key_mapping[f"dec_blocks.{i}.norm2.weight"] = f"blocks.{i}.norm2.weight"
            key_mapping[f"dec_blocks.{i}.norm2.bias"] = f"blocks.{i}.norm2.bias"
            key_mapping[f"dec_blocks.{i}.norm3.weight"] = f"blocks.{i}.norm3.weight"
            key_mapping[f"dec_blocks.{i}.norm3.bias"] = f"blocks.{i}.norm3.bias"
            key_mapping[f"dec_blocks.{i}.norm_y.weight"] = f"blocks.{i}.norm_y.weight"
            key_mapping[f"dec_blocks.{i}.norm_y.bias"] = f"blocks.{i}.norm_y.bias"

            # MLP layers
            key_mapping[f"dec_blocks.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
            key_mapping[f"dec_blocks.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
            key_mapping[f"dec_blocks.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
            key_mapping[f"dec_blocks.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"

        # Create new state dict with mapped keys
        # Get all keys from both decoders
        annotated_keys = set(annotated_decoder.state_dict().keys())
        croco_mapped_keys = set(key_mapping.values())
        croco_original_keys = set(key_mapping.keys())
        croco_actual_keys = set(croco.state_dict().keys())

        # Find keys that exist in annotated but not mapped from croco
        keys_missing_in_mapping = annotated_keys - croco_mapped_keys
        if keys_missing_in_mapping:
            print("\nKeys in annotated decoder but not mapped from croco:")
            print(keys_missing_in_mapping)

        # Find mapped keys that don't exist in annotated
        invalid_mapped_keys = croco_mapped_keys - annotated_keys
        if invalid_mapped_keys:
            print("\nMapped keys that don't exist in annotated decoder:")
            print(invalid_mapped_keys)

        # Find croco keys in mapping that don't exist in croco
        nonexistent_croco_keys = croco_original_keys - croco_actual_keys
        if nonexistent_croco_keys:
            print("\nKeys in mapping that don't exist in croco:")
            print(nonexistent_croco_keys)

        # Find croco keys that aren't mapped
        unmapped_croco_keys = croco_actual_keys - croco_original_keys
        if unmapped_croco_keys:
            print("\nKeys in croco that aren't mapped:")
            print(unmapped_croco_keys)

        new_state_dict = {}
        for croco_key, annotated_key in key_mapping.items():
            new_state_dict[annotated_key] = croco_state_dict[croco_key]

        # Load the mapped state dict into annotated decoder
        missing_keys, unexpected_keys = annotated_decoder.load_state_dict(new_state_dict, strict=False)
        print(f"\nMissing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # Verify decoder block weights are identical
        print("\nVerifying decoder block weights...")
        croco_state_dict = croco.state_dict()
        annotated_state_dict = annotated_decoder.state_dict()

        # Construct inverse key mapping for verification
        inverse_key_mapping = {v: k for k, v in key_mapping.items()}

        for key, value in annotated_state_dict.items():
            croco_key = inverse_key_mapping[key]
            assert torch.allclose(croco_state_dict[croco_key], value), f"Weights differ for {croco_key} vs {key}"

        print("All decoder block weights verified to be identical")

    # Forward pass
    with torch.no_grad():
        print("\nGenerating encoder outputs...")
        torch.manual_seed(42)
        masked_image_tokens, masked_image_pos, masked_image_mask = croco._encode_image(x, do_mask=True)
        reference_image_tokens, reference_image_pos, _ = croco._encode_image(x, do_mask=False)

        torch.manual_seed(42)
        print("\nCroCo decoder forward pass...")
        croco_block_outputs = croco._decoder(
            masked_image_tokens,
            masked_image_pos,
            masked_image_mask,
            reference_image_tokens,
            reference_image_pos,
            return_all_blocks=True,
        )

        print("\nGenerating encoder outputs for annotated...")
        torch.manual_seed(42)
        masked_image_tokens_for_annotated, masked_image_pos_for_annotated, masked_image_mask_for_annotated = (
            croco._encode_image(x, do_mask=True)
        )
        reference_image_tokens_for_annotated, reference_image_pos_for_annotated, _ = croco._encode_image(
            x, do_mask=False
        )

        # Verify inputs are equivalent before decoder forward pass
        assert torch.allclose(masked_image_tokens, masked_image_tokens_for_annotated), "Masked image tokens differ"
        assert torch.allclose(masked_image_pos, masked_image_pos_for_annotated), "Masked image positions differ"
        assert torch.allclose(masked_image_mask, masked_image_mask_for_annotated), "Masked image masks differ"
        assert torch.allclose(
            reference_image_tokens, reference_image_tokens_for_annotated
        ), "Reference image tokens differ"
        assert torch.allclose(
            reference_image_pos, reference_image_pos_for_annotated
        ), "Reference image positions differ"

        torch.manual_seed(42)
        print("\nAnnotated decoder forward pass...")
        annotated_block_outputs = annotated_decoder(
            masked_image_tokens_for_annotated,
            masked_image_pos_for_annotated,
            masked_image_mask_for_annotated,
            reference_image_tokens_for_annotated,
            reference_image_pos_for_annotated,
            return_all_blocks=True,
        )

        print("\nComparing intermediate values...")
        # Compare decoder embeddings
        croco_decoder_embed = croco.decoder_embed(masked_image_tokens)
        annotated_decoder_embed = annotated_decoder.decoder_embed(masked_image_tokens_for_annotated)
        embed_max_diff = torch.max(torch.abs(croco_decoder_embed - annotated_decoder_embed)).item()
        print(f"Decoder embed max difference: {embed_max_diff:.2e}")

        # Compare block outputs
        print("\nComparing block outputs...")
        for i, (croco_block, annotated_block) in enumerate(zip(croco_block_outputs, annotated_block_outputs)):
            block_max_diff = torch.max(torch.abs(croco_block - annotated_block)).item()
            print(f"Block {i} max difference: {block_max_diff:.2e}")
            assert block_max_diff < 1e-6, f"Block {i} difference exceeds tolerance"

        # Compare positional embeddings
        print(f"CroCo dec_pos_embed shape: {croco.dec_pos_embed.shape}")
        print(f"Annotated dec_pos_embed shape: {annotated_decoder.dec_pos_embed.shape}")
        pos_max_diff = torch.max(torch.abs(croco.dec_pos_embed - annotated_decoder.dec_pos_embed)).item()
        print(f"Positional embedding max difference: {pos_max_diff:.2e}")

        # Compare mask tokens
        print(f"CroCo mask_token shape: {croco.mask_token.shape}")
        print(f"Annotated mask_token shape: {annotated_decoder.mask_token.shape}")
        mask_max_diff = torch.max(torch.abs(croco.mask_token - annotated_decoder.mask_token)).item()
        print(f"Mask token max difference: {mask_max_diff:.2e}")

    croco_dec_feat = croco_block_outputs[-1]
    annotated_dec_feat = annotated_block_outputs[-1]

    # Compare outputs
    max_diff = torch.max(torch.abs(annotated_dec_feat - croco_dec_feat)).item()
    mean_diff = torch.mean(torch.abs(annotated_dec_feat - croco_dec_feat)).item()

    print("\nDecoder Implementation Comparison:")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Mean difference: {mean_diff:.2e}")
    print(f"Shapes match: {annotated_dec_feat.shape == croco_dec_feat.shape}")
    print(f"Output shape: {annotated_dec_feat.shape}")

    # Compare outputs at each position
    if annotated_dec_feat.shape == croco_dec_feat.shape:
        B, N, D = annotated_dec_feat.shape
        for i in range(N):
            pos_diff = torch.max(torch.abs(annotated_dec_feat[:, i, :] - croco_dec_feat[:, i, :])).item()
            if pos_diff > 1e-6:
                print(f"Position {i} max difference: {pos_diff:.2e}")

    assert max_diff < 1e-6, "Maximum difference exceeds tolerance"
    assert mean_diff < 1e-6, "Mean difference exceeds tolerance"
    assert annotated_dec_feat.shape == croco_dec_feat.shape, "Output shapes do not match"
