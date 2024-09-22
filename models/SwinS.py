import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

class PatchEmbedding(nn.Module):
    """
    This class implements the Patch Embedding mechanism, where the input image is divided into patches, and each patch is treated as an embedding.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        patch_size (int): Size of each patch.
        C (int): Output embedding dimension.

    """
    def __init__(self, in_channels, patch_size, C):
        super().__init__()
        
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.C = C

        # Convolution for patch embedding (downsampling by patch_size).
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=C, 
            kernel_size=patch_size, 
            stride=patch_size,
        )

        self.layer_norm = nn.LayerNorm(normalized_shape=C)
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of PatchEmbedding.

        Args:
            x (Tensor): Input image of shape [B, C_in, H, W].

        Returns:
            Tensor: Output of shape [B, (H//patch_size)*(W//patch_size), C].
        """
        # Shape: [B, C, H//patch_size, W//patch_size]
        x = self.conv(x)
        
        # Flatten spatial dimensions: [B, C, H', W'] -> [B, H'*W', C]
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Apply layer normalization.
        x = self.layer_norm(x)
        
        # Apply ReLU activation.
        x = self.act(x)

        return x

class PatchMerging(nn.Module):
    """
    Patch Merging layer which reduces the spatial dimensions while increasing the channel dimensions.
    
    Args:
        C (int): Input dimension (embedding size of patches).
    """
    def __init__(self, C):
        super().__init__()
        self.C = C
        self.linear = nn.Linear(in_features=4*C, out_features=2*C)
        self.layer_norm = nn.LayerNorm(normalized_shape=2*C)

    def forward(self, x):
        """
        Forward pass of PatchMerging.

        Args:
            x (Tensor): Input tensor of shape [B, H*W, C].

        Returns:
            Tensor: Output tensor of shape [B, (H//2)*(W//2), 2*C].
        """
        # Calculate new spatial dimensions after patch merging.
        h = w = int(math.sqrt(x.shape[1]) / 2)
        
        # Reshape into 2x2 patches: [B, H*W, C] -> [B, (H//2)*(W//2), 4*C]
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=h, w=w)
        
        # Linear transformation to increase channels: 4*C -> 2*C.
        x = self.linear(x)
        
        # Apply layer normalization.
        x = self.layer_norm(x)

        return x

class RelativeEmbeddings(nn.Module):
    """
    Implements relative positional embeddings for the attention mechanism.

    Args:
        window_size (int): Size of the window in the shifted window MSA.
    """
    def __init__(self, window_size):
        super().__init__()

        B = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        x = torch.arange(1, window_size + 1, 1 / window_size)
        x = (x[None, :]-x[:, None]).int()
        y = torch.concat([torch.arange(1, window_size + 1)] * window_size)
        y = (y[None, :]-y[:, None])
        
        # Relative positional embeddings.
        self.embeddings = nn.Parameter((B[x[:, :], y[:, :]]), requires_grad = False)

    def forward(self, x):
        """
        Forward pass to add relative positional embeddings.

        Args:
            x (Tensor): Input tensor of shape [B, heads, window_size*window_size, window_size*window_size].

        Returns:
            Tensor: Output tensor of shape [B, heads, window_size*window_size, window_size*window_size].
        """
        return x + self.embeddings

class ShiftedWindowMSA(nn.Module):
    """
    Implements the Shifted Window Multi-head Self-Attention (MSA) mechanism.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mask (bool): Whether to use masking for shifted windows.
    """
    def __init__(self, embed_dims, num_heads, window_size, mask=False):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dims // num_heads
        self.mask = mask
        self.embeddings = RelativeEmbeddings(window_size)

        self.scale = 1 / (self.head_dim ** 0.5)
        
        # Linear layer for generating query, key, and value.
        self.qkv = nn.Linear(in_features=embed_dims, out_features=3*embed_dims)
        
        # Output projection layer.
        self.projection2 = nn.Linear(in_features=embed_dims, out_features=embed_dims)

    def forward(self, x):
        """
        Forward pass of Shifted Window MSA.

        Args:
            x (Tensor): Input tensor of shape [B, H*W, C].

        Returns:
            Tensor: Output tensor of shape [B, H*W, C].
        """
        # Calculate the spatial dimensions (H, W) from the flattened patch embeddings.
        h = w = int(math.sqrt(x.shape[1]))

        # Get query, key, value: [B, H*W, 3*C]
        qkv = self.qkv(x)

        # Reshape into windowed form: [B, H', W', num_heads, head_dim, 3]
        qkv = rearrange(qkv, 'b (h w) (c K) -> b h w c K', h=h, w=w, K=3)
        qkv = rearrange(qkv, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K', 
                        H=self.num_heads, m1=self.window_size, m2=self.window_size)

        # Apply window shifting if mask is enabled.
        if self.mask:
            x = torch.roll(qkv, (-self.window_size//2, -self.window_size//2), dims=(1, 2))

        # Split into query, key, value.
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = q.squeeze(-1), k.squeeze(-1), v.squeeze(-1)

        # Transpose for matrix multiplication: k -> [B, heads, H', W', head_dim].
        k_t = k.transpose(-2, -1)

        # Compute attention scores: [B, heads, H', W', H'W'].
        attention_scores = (q @ k_t) * self.scale
        
        # Add relative positional embeddings.
        attention_scores = self.embeddings(attention_scores)

        # Apply masking if enabled.
        if self.mask:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1=self.window_size, w2=self.window_size).cuda()
            attention_scores[:, :, -1, :] += row_mask
            attention_scores[:, :, :, -1] += column_mask

        # Compute attention: [B, heads, H', W', head_dim].
        attention = F.softmax(attention_scores, dim=-1) @ v

        # Reshape back to original dimensions: [B, H', W', num_heads*head_dim].
        x = rearrange(attention, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', 
                      H=self.num_heads, m1=self.window_size, m2=self.window_size)
        
        if self.mask:
            # Reverse window shift after MSA.
            x = torch.roll(x, (self.window_size//2, self.window_size//2), dims=(1, 2))
        
        # Flatten the spatial dimensions again.
        x = rearrange(x, 'b h w c -> b (h w) c')
        
        # Apply the final output projection.
        x = self.projection2(x)

        return x

class SwinTransformerBlock(nn.Module):
    """
    Implements a Swin Transformer Block with shifted window MSA and MLP layers.

    Args:
        embed_dims (int): Input embedding dimensions.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        mask (bool): Whether to apply window shifting in the first MSA.
    """
    def __init__(self, embed_dims, num_heads, window_size, mask):
        super().__init__()

        self.attention = ShiftedWindowMSA(embed_dims, num_heads, window_size, mask)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embed_dims, out_features=4*embed_dims),
            nn.GELU(),
            nn.Linear(in_features=4*embed_dims, out_features=embed_dims),
        )
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dims)

    def forward(self, x):
        """
        Forward pass of the Swin Transformer Block.

        Args:
            x (Tensor): Input tensor of shape [B, H*W, C].

        Returns:
            Tensor: Output tensor of shape [B, H*W, C].
        """
        # First attention layer.
        residual = x
        x = self.layer_norm(x)
        x = self.attention(x)
        x = x + residual

        # MLP layer.
        residual = x
        x = self.layer_norm(x)
        x = self.mlp(x)
        x = x + residual

        return x

class AlternatingAttentionBlock(nn.Module):
    """
    Implements an Alternating Attention Block that applies two attention mechanisms in sequence:
    1. **Window-based Self-Attention (WSA)** without shifting windows.
    2. **Shifted Window Multi-head Self-Attention (MSA)** with window shifting.

    This structure alternates between these two types of attention to capture both local and global dependencies
    across the input sequence of patches.

    Args:
        embed_dims (int): Dimension of the input embeddings (features).
        num_heads (int): Number of attention heads in multi-head self-attention.
        window_size (int): Size of the attention window.

    """
    def __init__(self, embed_dims, num_heads, window_size):
        super().__init__()
        # Window-based Self-Attention (WSA) without window shifting.
        self.WSA = SwinTransformerBlock(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, mask=False)
        
        # Shifted Window Multi-head Self-Attention (MSA) with window shifting.
        self.MSA = ShiftedWindowMSA(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, mask=True)

    def forward(self, x):
        """
        Forward pass of the Alternating Attention Block.

        First applies Window-based Self-Attention (WSA), followed by Shifted Window Multi-head Self-Attention (MSA).

        Args:
            x (Tensor): Input tensor of shape [B, H*W, C], where:
                B is the batch size,
                H*W is the number of patches,
                C is the embedding dimension (feature size).

        Returns:
            Tensor: Output tensor of shape [B, H*W, C] after applying both WSA and MSA.
        """
        # Apply Window-based Self-Attention (WSA) without shifting.
        out = self.WSA(x)
        
        # Apply Shifted Window Multi-head Self-Attention (MSA) with shifting.
        out = self.MSA(out)

        return out

class SwinTransformerS(nn.Module):
    """
    Implements the Swin Transformer architecture for image classification.

    This model is based on hierarchical feature representation through alternating attention mechanisms and patch merging. 
    It processes images using a window-based self-attention mechanism that shifts the window in alternating layers to capture both 
    local and global dependencies efficiently.

    Args:
        in_channels (int, optional): Number of input image channels. Default is 3 (RGB image).
        num_classes (int, optional): Number of output classes for classification. Default is 1000.
        embed_dims (int, optional): Number of embedding dimensions for the input patches. Default is 96.
        num_heads (int, optional): Number of attention heads for multi-head self-attention. Default is 3.
        window_size (int, optional): Size of the attention window. Default is 7.
        stage_repeats (list, optional): Number of alternating attention blocks for each of the four stages. 
                                        It should be a list of four integers. Default is [2, 2, 18, 2].

    """
    def __init__(self, in_channels=3, num_classes=1000, embed_dims=96, num_heads=32, window_size=7, stage_repeats=[2, 2, 18, 2]):
        super().__init__()

        # Embedding layer to convert input image into patch embeddings.
        self.embedding = PatchEmbedding(in_channels=in_channels, patch_size=4, C=embed_dims)

        # First patch merging layer after stage 1.
        self.patch_merging1 = PatchMerging(C=embed_dims)

        # Stage 1 consisting of alternating attention blocks.
        self.stage1 = nn.Sequential(
            *[AlternatingAttentionBlock(embed_dims=embed_dims, num_heads=num_heads, window_size=window_size) for _ in range(stage_repeats[0] // 2)]
        )

        # Second patch merging layer after stage 2.
        self.patch_merging2 = PatchMerging(C=embed_dims * 2)

        # Stage 2 with doubled embedding dimension and number of heads.
        self.stage2 = nn.Sequential(
            *[AlternatingAttentionBlock(embed_dims=embed_dims * 2, num_heads=num_heads, window_size=window_size) for _ in range(stage_repeats[1] // 2)]
        )

        # Third patch merging layer after stage 3.
        self.patch_merging3 = PatchMerging(C=embed_dims * 4)

        # Stage 3 with further increased embedding dimension and number of heads.
        self.stage3 = nn.Sequential(
            *[AlternatingAttentionBlock(embed_dims=embed_dims * 4, num_heads=num_heads, window_size=window_size) for _ in range(stage_repeats[2] // 2)]
        )

        # Final stage without patch merging.
        self.stage4 = nn.Sequential(
            *[AlternatingAttentionBlock(embed_dims=embed_dims * 8, num_heads=num_heads, window_size=window_size) for _ in range(stage_repeats[3] // 2)]
        )

        # Layer normalization for the final stage.
        self.layer_norm = nn.LayerNorm(normalized_shape=embed_dims * 8)

        # Global average pooling to reduce the dimensionality for classification.
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        # Fully connected layer for final classification output.
        self.fc = nn.Linear(in_features=embed_dims * 8, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the Swin Transformer.

        Args:
            x (Tensor): Input tensor of shape [B, C, H, W], where:
                B is the batch size,
                C is the number of input channels (default: 3),
                H is the height of the image,
                W is the width of the image.

        Returns:
            Tensor: Output tensor of shape [B, num_classes], representing the class scores for each input image.
        """
        # Convert input image into patch embeddings.
        x = self.embedding(x)

        # Apply Stage 1 alternating attention blocks.
        x = self.stage1(x)

        # Patch merging and move to Stage 2.
        x = self.patch_merging1(x)
        x = self.stage2(x)

        # Patch merging and move to Stage 3.
        x = self.patch_merging2(x)
        x = self.stage3(x)

        # Patch merging and move to Stage 4.
        x = self.patch_merging3(x)
        x = self.stage4(x)

        # Apply layer normalization.
        x = self.layer_norm(x)

        # Transpose for global average pooling.
        x = x.transpose(1, 2)

        # Perform adaptive average pooling to get a fixed-size representation.
        x = self.avgpool(x)

        # Flatten the output tensor and apply the fully connected layer for classification.
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary

    model = SwinTransformerS().to('cuda')
    print(summary(model, (3, 224, 224)))