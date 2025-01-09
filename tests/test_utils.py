from typing import Dict, List, Set, Tuple

import torch


def load_and_validate_state_dict_with_mapping(
    annotated_model,
    croco_model,
    key_mapping: Dict[str, str],
    print_debug: bool = True,
    strict_mapping: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Validate state dict mapping between annotated and croco models.

    Args:
        annotated_model: The annotated model instance
        croco_model: The croco model instance
        key_mapping: Dict mapping from croco keys to annotated keys
        print_debug: Whether to print debug information

    Returns:
        new_state_dict: The mapped state dict ready to load into annotated model

    Raises:
        AssertionError: If there are any validation errors in the mapping
    """
    errors: List[str] = []

    # Get all keys from both models
    annotated_keys = set(annotated_model.state_dict().keys())
    croco_mapped_keys = set(key_mapping.values())
    croco_original_keys = set(key_mapping.keys())
    croco_actual_keys = set(croco_model.state_dict().keys())

    # Find keys that exist in annotated but not mapped from croco
    keys_missing_in_mapping = annotated_keys - croco_mapped_keys
    if keys_missing_in_mapping:
        errors.append(f"Keys in annotated model but not mapped from croco: {keys_missing_in_mapping}")

    # Find mapped keys that don't exist in annotated
    invalid_mapped_keys = croco_mapped_keys - annotated_keys
    if invalid_mapped_keys:
        errors.append(f"Mapped keys that don't exist in annotated model: {invalid_mapped_keys}")

    # Find croco keys in mapping that don't exist in croco
    nonexistent_croco_keys = croco_original_keys - croco_actual_keys
    if nonexistent_croco_keys:
        errors.append(f"Keys in mapping that don't exist in croco: {nonexistent_croco_keys}")

    # Find croco keys that aren't mapped
    unmapped_croco_keys = croco_actual_keys - croco_original_keys
    if unmapped_croco_keys and strict_mapping:
        errors.append(f"Keys in croco that aren't mapped: {unmapped_croco_keys}")

    # Create new state dict with mapped keys
    croco_state_dict = croco_model.state_dict()
    new_state_dict = {}
    for croco_key, annotated_key in key_mapping.items():
        if croco_key in croco_state_dict:
            new_state_dict[annotated_key] = croco_state_dict[croco_key]

    # Load and verify the mapped state dict
    missing_keys, unexpected_keys = annotated_model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        errors.append(f"Missing keys when loading state dict: {missing_keys}")
    if unexpected_keys:
        errors.append(f"Unexpected keys when loading state dict: {unexpected_keys}")

    if print_debug:
        print("\nVerifying model weights...")

    # Verify weights are identical
    croco_state_dict = croco_model.state_dict()
    annotated_state_dict = annotated_model.state_dict()

    # Construct inverse key mapping for verification
    inverse_key_mapping = {v: k for k, v in key_mapping.items()}

    for key, value in annotated_state_dict.items():
        if key in inverse_key_mapping:
            croco_key = inverse_key_mapping[key]
            if not torch.allclose(croco_state_dict[croco_key], value):
                errors.append(f"Weights differ for {croco_key} vs {key}")

    if print_debug and not errors:
        print("All weights verified to be identical")

    # Raise assertion error if any validation errors occurred
    assert not errors, "\n".join(errors)
