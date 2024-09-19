import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Layer: Converts an image into patch embeddings.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB images).
        embed_dims (int): The dimensionality of the patch embeddings.
        patch_size (int): The size of each square patch.
        img_size (int): The size of the input image (assuming a square image).
    """
    def __init__(self, in_channels, embed_dims, patch_size, img_size):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        
        # Calculate the number of patches in the image.
        self.num_patches = ((img_size * img_size) // (patch_size ** 2))

        # Convolution layer to convert image patches into embeddings.
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dims, 
            kernel_size=patch_size,
            stride=patch_size
        )

        # Flatten the patches into a sequence of tokens.
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

    def forward(self, x):
        """
        Forward pass for the patch embedding.
        
        Args:
            x (torch.Tensor): Input image of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Flattened patch embeddings of shape (batch_size, num_patches, embed_dims).
        """
        # Apply the convolution and rearrange the dimensions.
        x = self.conv(x).permute((0, 2, 3, 1))  # (BS, C, H, W) -> (BS, embed_dims, H, W) -> (BS, H, W, embed_dims)
        
        # Flatten the patch dimensions into a sequence of patches.
        x = self.flatten(x)  # (BS, H, W, embed_dims) -> (BS, num_patches, embed_dims)
       
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Layer: Computes attention over tokens.

    Args:
        embed_dims (int): The dimensionality of the input tokens.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, embed_dims, num_heads):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        
        # The dimensionality of each attention head.
        self.head_dim = embed_dims // num_heads
        # Scaling factor for the dot-product attention.
        self.scale = 1 / (self.head_dim ** 0.5)

        # Linear layers to compute query, key, and value (all packed into one).
        self.qkv = nn.Linear(in_features=embed_dims, out_features=embed_dims * 3)
        # Linear projection after attention.
        self.projection = nn.Linear(in_features=embed_dims, out_features=embed_dims)

    def forward(self, x):
        """
        Forward pass for the self-attention layer.
        
        Args:
            x (torch.Tensor): Input token embeddings of shape (batch_size, num_tokens, embed_dims).
        
        Returns:
            torch.Tensor: Output after self-attention, shape (batch_size, num_tokens, embed_dims).
        """
        batch_size, num_tokens, embed_dims = x.shape  # num_tokens = num_patches + 1 (includes class token)

        if embed_dims != self.embed_dims:
            raise ValueError("Embedding dimensions mismatch!")

        # Compute query, key, and value for all tokens.
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)  # Split into q, k, v.
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Re-arrange dimensions to separate q, k, v. (3, batch_size, num_heads, num_patches + 1, head_dim) 
        q, k, v = qkv[0], qkv[1], qkv[2]  # Unpack q, k, v.

        # Transpose key for dot-product attention.
        k_t = torch.transpose(k, dim0=-2, dim1=-1)  # Transpose to match q's dimensions. k: (batch_size, num_heads, num_patches + 1, head_dim) -> k_t: (batch_size, num_heads, head_dim, num_patches + 1)
        
        # Compute the scaled dot-product attention.
        dot_prod = (q @ k_t) * self.scale  # (batch_size, num_heads, num_patches + 1, head_dim) x (batch_size, num_heads, head_dim, num_patches + 1) * 1 / sqrt(head_dim) -> (batch_size, num_heads, num_patches + 1, num_patches + 1)
        dot_prod = torch.softmax(dot_prod, dim=-1)  # Normalize with softmax. These will act as the weights to be applied to the values vector 'v'.

        # Compute the weighted sum over value vectors.
        weighted_values = dot_prod @ v  # (batch_size, num_heads, num_patches + 1, num_patches + 1) x (batch_size, num_heads, num_patches + 1, head_dim) -> (batch_size, num_heads, num_patches + 1, head_dim)
        
        # Reshape back to (batch_size, num_patches + 1, embed_dims).
        weighted_values = weighted_values.permute(0, 2, 1, 3).flatten(start_dim=2, end_dim=3)
        
        # Project the attention output back to embed_dims.
        x = self.projection(weighted_values)

        return x

class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used after self-attention.

    Args:
        in_features (int): Input dimensionality.
        mlp_size (int): Hidden layer size.
        out_features (int): Output dimensionality.
        dropout_p (float): Dropout probability.
    """
    def __init__(self, in_features, mlp_size, out_features, dropout_p):
        super().__init__()

        self.fc1 = nn.Linear(in_features=in_features, out_features=mlp_size)
        self.act = nn.GELU()  # Activation function.
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(in_features=mlp_size, out_features=out_features)

    def forward(self, x):
        """
        Forward pass for the MLP block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_patches, in_features).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_patches, out_features).
        """
        x = self.fc1(x)
        x = self.act(x)  # GELU activation.
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder Block: Combines multi-head self-attention and MLP.

    Args:
        embed_dims (int): Embedding dimensionality.
        num_heads (int): Number of attention heads.
        mlp_size (int): Size of the hidden layer in the MLP.
        dropout_p (float): Dropout probability.
    """
    def __init__(self, embed_dims, num_heads, mlp_size, dropout_p):
        super().__init__()

        # Layer normalization.
        self.norm = nn.LayerNorm(embed_dims)
        # Multi-head self-attention.
        self.msa = MultiHeadSelfAttention(embed_dims=embed_dims, num_heads=num_heads)
        # Multi-layer perceptron.
        self.mlp = MultiLayerPerceptron(in_features=embed_dims, mlp_size=mlp_size, out_features=embed_dims, dropout_p=dropout_p)

    def forward(self, x):
        """
        Forward pass for the transformer encoder block.
        
        Args:
            x (torch.Tensor): Input token embeddings of shape (batch_size, num_patches, embed_dims).
        
        Returns:
            torch.Tensor: Output after transformer block of shape (batch_size, num_patches, embed_dims).
        """
        out = self.norm(x)
        out = self.msa(out)
        out += x  # Residual connection.
        
        x = out
        out = self.norm(out)
        out = self.mlp(out)
        out += x  # Residual connection.

        return out

class ViTHuge(nn.Module):
    """
    Vision Transformer (ViT) Model: Implementation of the huge ViT architecture.

    Args:
        in_channels (int): Number of input channels (3 for RGB).
        num_classes (int): Number of output classes for classification.
        num_layers (int): Number of transformer encoder layers.
        embed_dims (int): Dimensionality of the embeddings.
        mlp_size (int): Size of the MLP hidden layer.
        num_heads (int): Number of attention heads.
        dropout_p (float): Dropout probability.
        patch_size (int): Size of image patches.
        img_size (int): Size of the input image (height or width).
    """
    def __init__(self, in_channels=3, num_classes=1000, num_layers=32, embed_dims=1280, mlp_size=5120, num_heads=16, dropout_p=0.1, patch_size=16, img_size=224):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, embed_dims, patch_size, img_size)
        
        # Positional embeddings.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, embed_dims))
        self.dropout = nn.Dropout(dropout_p)

        # Stacking multiple transformer encoder layers.
        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(embed_dims, num_heads, mlp_size, dropout_p) for _ in range(num_layers)
        ])

        # Final classification head.
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for the Vision Transformer (ViT).
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, in_channels, height, width).
        
        Returns:
            torch.Tensor: Logits for each class (batch_size, num_classes).
        """
        batch_size = x.shape[0]

        # Patch embedding.
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dims)

        # Append class token and add positional embeddings.
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # Class token for each sample in the batch.
        x = torch.cat([cls_token, x], dim=1)  # (batch_size, 1 + num_patches, embed_dims)
        x = x + self.pos_embed  # Add positional embeddings.
        x = self.dropout(x)

        # Pass through each transformer encoder layer.
        for layer in self.transformer_layers:
            x = layer(x)

        # Take the class token representation (the first token).
        x = x[:, 0]

        # Classification head.
        x = self.mlp_head(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary

    model = ViTHuge().to('cuda')
    print(summary(model, (3, 224, 224)))