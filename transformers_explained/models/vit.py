"""Classification backbone using vit."""
import torch
from torch import nn

class PatchEmbedding(nn.Module):
    """Patch Embedding Layer for vision transformer.
    
    Args:
        in_channels: Number of input channels.
        patch_size: Size of each patch.
        emb_size: Size of embeddings.
        image_size: Size of the input images.
    """

    def __init__(self, in_channels: int, patch_size: int, emb_size: int, image_size: int) -> None:
        """ Initialization routene. """

        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=emb_size, kernel_size=patch_size, stride=patch_size)
        self.bn = nn.BatchNorm2d(emb_size)


    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass for the patch embedding layer.
        
        Args:
            in_tensor: Input tensor.
        
        Returns:
            Output tensor with patches embedded.
        """

        out_tensor = self.proj(in_tensor)  # (B, E, H/P, W/P)
        out_tensor = self.bn(out_tensor)
        out_tensor = out_tensor.flatten(2)  # (B, E, N)
        out_tensor = out_tensor.transpose(1, 2)  # (B, N, E)
        return out_tensor

class VisionTransformerBackbone(nn.Module):
    """Vision Transformer Back Bone.
    
    Args:
        img_size: Size of the input image.
        patch_size: Size of each patch.
        emb_size: Embedding size.
        depth: Number of transformer encoder layers.
        num_heads: Number of attention heads in each transformer encoder layer.
        dropout_rate: Dropout rate for regularization.
    """

    def __init__(self, image_size: int, patch_size: int, emb_size: int, depth: int,
                  num_heads: int, dropout_rate: float = 0.1) -> None:
        """Initialization routine."""
        super().__init__()
        self.patch_embeddings = PatchEmbedding(in_channels=3, patch_size=patch_size, emb_size=emb_size, image_size=image_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, (image_size // patch_size) ** 2, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=depth)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weights = []


    def save_attention_weights(self, module, input, output):
        """Save attention weights from the multi-head attention layer."""
        attn_output_weights = module.self_attn.attn_output_weights
        if attn_output_weights is not None:
            self.attention_weights.append(attn_output_weights.detach().cpu().numpy())


    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        """Forward Pass for vision transformer back bone.

        Args:
            in_tensor: Input tensor.
        
        Returns: 
            Output tensor after passing through the network.
        """

        self.attention_weights = []  # Reset attention weights
        out_tensor = self.patch_embeddings(in_tensor) + self.positional_encoding    
        handles = []    
        for layer in self.transformer_encoder.layers:
            handle = layer.self_attn.register_forward_hook(self.save_attention_weights)
            handles.append(handle)
         
        out_tensor = self.transformer_encoder(out_tensor)
        
        for handle in handles:
            handle.remove()
        
        out_tensor = out_tensor.mean(dim=1)  # Global average pooling
        out_tensor = self.dropout(out_tensor)
        return out_tensor


    def get_attention_map(self) -> list:
        """Get the stored attention maps.
        
        Returns:
            List of attention maps for each layer.
        """
        return self.attention_weights
