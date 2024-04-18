import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768
                 ):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=1,
                                  end_dim=2)

    def forward(self, x):
        image_resolution = x.shape[-1]
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched.permute((0,3,2,1)))

        return x_flattened


class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_patches):
        super(PositionalEmbedding, self).__init__()
        position = torch.arange(0,num_patches, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embedding_dim, 2) * -(torch.log(torch.tensor(10000)) / embedding_dim))
        positionalencoding = nn.Parameter(torch.zeros(num_patches + 1, embedding_dim))
        positionalencoding[:, 0::2].data = torch.sin(position * divisor)
        positionalencoding[:, 1::2].data = torch.cos(position * divisor)

        self.positional_embeddings = nn.Parameter(positionalencoding, requires_grad=False)

    def forward(self, x):
        return   x + self.positional_embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int = 768,
                 attn_dropout: float = 0, num_heads: int = 16):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_layer = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                     dropout=attn_dropout,
                                                     num_heads=num_heads)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_layer(query=x,
                                              key=x,
                                              value=x,
                                              need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int = 768,
                 mlp_size: int = 3072,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp_layer = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp_layer(x)

        return x


class TransformerBlock(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 16,
                 mlp_dropout: float = 0.1,
                 mlp_size: int = 3072,
                 attn_dropout: float = 0.0):
        super().__init__()
        self.msa_block = MultiHeadSelfAttention(embedding_dim=embedding_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x


class VIT(nn.Module):
    def __init__(self,
                 embedding_dim: int = 768,
                 num_heads: int = 12,
                 mlp_dropout: float = 0.1,
                 mlp_size: int = 3072,
                 attn_dropout: float = 0.0,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 img_size: int = 224,
                 num_transformer_layer: int = 6,
                 num_class: int = 1000
                 ):
        super().__init__()
        self.num_patches = (img_size * img_size) // (patch_size ** 2)
        self.patched_images = PatchEmbedding(in_channels=in_channels,
                                             patch_size=patch_size,
                                             embedding_dim=embedding_dim)
        self.position_embedding = PositionalEmbedding(embedding_dim=embedding_dim,
                                                      num_patches=self.num_patches)
        self.transformer_block = nn.Sequential(*[TransformerBlock(attn_dropout=attn_dropout,
                                                                  mlp_size=mlp_size,
                                                                  mlp_dropout=mlp_dropout,
                                                                  num_heads=num_heads,
                                                                  embedding_dim=embedding_dim) for _ in
                                                 range(num_transformer_layer)])
        assert img_size % patch_size == 0
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)
        self.num_patches = (img_size * img_size) // (patch_size ** 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_class)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patched_images(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding(x)
        x = self.transformer_block(x)
        x = self.classifier(x[:, 0])

        return x
