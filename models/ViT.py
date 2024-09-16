import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)  # (num_samples, embed_dim, num_patches ** 0.5, numpatches ** 0.5)
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, qkv_bias, attention_dropout_p, projection_dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(p=attention_dropout_p)
        self.projection = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.projection_dropout = nn.Dropout(p=projection_dropout_p)

    def forward(self, x):
        num_samples, num_tokens, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError

        qkv = self.qkv(x)
        qkv = qkv.reshape(num_samples, num_tokens, 3, self.num_heads, self.head_dim)  # (num_samples, num_patches + 1, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, num_samples, num_heads, num_patches + 1, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)  # (num_samples, num_heads, head_dim, num_patches + 1)
        dp = (q @ k_t) * self.scale  # (num_samples, num_heads, num_patches + 1, num_patches + 1)
        attn = dp.softmax(dim=-1)  # (num_samples, num_heads, num_patches + 1, num_patches + 1)
        attn = self.attention_dropout(attn)

        weighted_avg = attn @ v  # (num_samples, num_heads, num_patches + 1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (num_samples, num_patches + 1, num_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (num_samples, num_patches + 1, embed_dim)

        x = self.projection(weighted_avg)  # (num_samples, num_patches + 1, embed_dim)
        x = self.projection_dropout(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.dropout = nn.Dropout(p) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True, attention_dropout_p=0., projection_dropout_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.attn = Attention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attention_dropout_p=attention_dropout_p, projection_dropout_p=projection_dropout_p)
        self.mlp = MLP(in_features=embed_dim, hidden_features=int(embed_dim * mlp_ratio), out_features=embed_dim, p=projection_dropout_p)   

    def forward(self, x):
        x = self.norm1(x)
        x += self.attn(x)
        x = self.norm2(x)
        x += self.mlp(x)
        return x

class ViT(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, qkv_bias=True, attention_dropout_p=0., projection_dropout_p=0., num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.position_dropout = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, attention_dropout_p=attention_dropout_p, projection_dropout_p=projection_dropout_p)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6)
        self.head = nn.Linear(in_features=embed_dim, out_features=num_classes)

    def forward(self, x):
        num_samples = x.shape[0]
        x = self.patch_embed(x)

        class_token = self.class_token.expand(num_samples, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x += self.position_embed
        x = self.position_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x