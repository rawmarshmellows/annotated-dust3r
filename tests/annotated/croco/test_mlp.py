import torch
import torch.nn as nn
from test_utils import load_and_validate_state_dict_with_mapping

from src.annotated.croco.mlp import Mlp as AnnotatedMLP
from src.croco.models.blocks import Mlp as CrocoMLP


def test_mlp_initialization():
    """Test initialization with different parameters"""
    in_features = 768
    hidden_features = 3072
    out_features = 768

    # Test basic initialization
    mlp = AnnotatedMLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    assert mlp.in_features == in_features
    assert mlp.hidden_features == hidden_features
    assert mlp.out_features == out_features

    # Verify layer dimensions
    assert isinstance(mlp.fc1, nn.Linear)
    assert isinstance(mlp.fc2, nn.Linear)
    assert mlp.fc1.in_features == in_features
    assert mlp.fc1.out_features == hidden_features
    assert mlp.fc2.in_features == hidden_features
    assert mlp.fc2.out_features == out_features


def test_mlp_forward():
    """Test forward pass and output shapes"""
    batch_size = 4
    seq_len = 196
    in_features = 768
    hidden_features = 3072
    out_features = 768

    mlp = AnnotatedMLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, in_features)

    # Test forward pass
    output = mlp(x)

    # Check output shape
    assert output.shape == (batch_size, seq_len, out_features)

    # Check output is different from input (transformation occurred)
    assert not torch.allclose(output, x)


def test_mlp_equivalence():
    """Test equivalence between Annotated and CroCo MLP implementations"""
    batch_size = 4
    seq_len = 196
    in_features = 768
    hidden_features = 3072
    out_features = 768

    # Initialize both implementations
    annotated_mlp = AnnotatedMLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    croco_mlp = CrocoMLP(in_features=in_features, hidden_features=hidden_features, out_features=out_features)

    # Initialize weights identically
    key_mapping = {
        "fc1.weight": "fc1.weight",
        "fc1.bias": "fc1.bias",
        "fc2.weight": "fc2.weight",
        "fc2.bias": "fc2.bias",
    }

    load_and_validate_state_dict_with_mapping(annotated_mlp, croco_mlp, key_mapping)

    # Create input tensor
    x = torch.randn(batch_size, seq_len, in_features)

    # Test forward pass
    with torch.no_grad():
        annotated_out = annotated_mlp(x)
        croco_out = croco_mlp(x)

        # Test outputs are close
        assert torch.allclose(annotated_out, croco_out)
