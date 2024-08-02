"""Sample test module to test generic moel."""

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from models.generic_model import get_model

@pytest.fixture(name='config_fixture')
def fixture_config_fixture() -> DictConfig:
    """Fixture for model configuration."""

    return OmegaConf.create({
        'model': {
            'name': 'vit',
            'img_size': 32,
            'patch_size': 4,
            'emb_size': 128,
            'depth': 2,
            'num_heads': 4,
            'drop_out_rate': 0.1,  # Correct key here
            'num_classes': 10
        },
        'layers': {
            'classification_head': {
                'fc1_out_features': 256
            }
        }
    })

def test_vit_forward(config_fixture: DictConfig) -> None:
    """Test the forward pass of the Vision Transformer model."""

    # GIVEN a model configuration
    model = get_model(config_fixture)
    x = torch.randn(2, 3, 32, 32)  # Batch size of 2, 3 channels, 32x32 images
    
    # WHEN passing the input tensor through the model
    output = model(x)
    
    # THEN the output shape should match the expected number of classes
    assert output.shape == (2, 10)