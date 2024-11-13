from typing import Optional, Tuple
import torch
import torch.nn as nn

class ImageLinearSignalConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
        self.kwargs = kwargs

class ImageEmbeddingLayer(nn.Module):
    def __init__(self, config: ImageLinearSignalConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
    
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'
        )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_position = self.num_patches
        self.position_embedding = nn.Embedding(self.num_position, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_position).expand((1, -1)),
            persistent=False
        )
    def forward(self, pixel_values: torch.FloatTensor):
        _, _, height, width = pixel_values.shape # [Batch_size, Channels, Height, Width]
        # num_patches_H = height // patch_size, num_patches_W = weight // patch_size
        # [Batch_size, embed_dim, num_patches_H, num_patches_W]
        patch_embeds = self.patch_embedding(pixel_values)
        # [Batch_size, embed_dim, num_patches_H * num_patches_W]
        embeddings = patch_embeds.flatten(2)
        # [Batch_size, num_patches_H * num_patches_W, embed_dim]
        embeddings = embeddings.transpose(1,2)
        # Add position embeddings
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings
        
          

class ImageTransformerLayer(nn.Module):
    def __init__(self, config: ImageLinearSignalConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = ImageEmbeddingLayer(config)
        self.encoder = ImageEncoderLayer(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state

class ImageLinearSignalModel(nn.Module):
    def __init__(self, config):
        self.config = config
        self.image_model = ImageTransformerLayer(config)
        
    def forward(self, pixel_values) -> Tuple:
        return self.image_model(pixel_values=pixel_values)
    
