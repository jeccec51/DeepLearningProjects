"""Test module to unittest vit."""

import pytest
import torch
from models.vit import VisionTransformerBackbone

@pytest.fixture(name='vit_model_fixture')
def fixture_vit_model() -> VisionTransformerBackbone:
    """Fixture for Vision Transformer model."""
    return VisionTransformerBackbone(
        image_size=32,
        patch_size=4,
        emb_size=128,
        depth=2,
        num_heads=4,
        dropout_rate=0.1
    )

def test_vit_forward(vit_model_fixture: VisionTransformerBackbone) -> None:
    """Test the forward pass of the Vision Transformer backbone."""
    # GIVEN a Vision Transformer model and an input tensor
    input_tensor = torch.randn(2, 3, 32, 32)  # Batch size of 2, 3 channels, 32x32 images
    
    # WHEN passing the input tensor through the model
    output_tensor = vit_model_fixture(input_tensor)
    
    # THEN the output shape should match the expected embedding size
    assert output_tensor.shape == (2, 128)

def test_get_attention_map(vit_model_fixture: VisionTransformerBackbone) -> None:
    """Test the attention map extraction of the Vision Transformer backbone."""
    # GIVEN a Vision Transformer model and an input tensor
    input_tensor = torch.randn(2, 3, 32, 32)
    
    # WHEN passing the input tensor through the model
    _ = vit_model_fixture(input_tensor)
    attentions = vit_model_fixture.get_attention_map()
    
    # THEN the number and shape of attention maps should be correct
    assert len(attentions) == 2  # Depth of 2 means 2 attention maps
    assert attentions[0].shape == (64, 2, 2)  # 2 samples, 4 heads, 64 tokens, 64 tokens
