import torch
import torch.nn as nn

class ConvNeXtBlock(nn.Module):
    """
    A ConvNeXt block, consisting of depthwise convolution followed by pointwise 
    convolutions with an expansion factor for increased intermediate feature dimensionality.

    Args:
        in_channels (int): Number of input channels.
        expansion_factor (int): Expansion factor for intermediate feature dimension.
                                Default is 4.
    """
    def __init__(self, in_channels, expansion_factor=4):
        super().__init__()

        # Intermediate dimension based on expansion factor
        intermediate_dims = in_channels * expansion_factor

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                   kernel_size=7, padding=3, groups=in_channels)
        
        # Pointwise convolutions
        self.pointwise1 = nn.Conv2d(in_channels=in_channels, out_channels=intermediate_dims, 
                                    kernel_size=1, padding=0)
        self.pointwise2 = nn.Conv2d(in_channels=intermediate_dims, out_channels=in_channels, 
                                    kernel_size=1, padding=0)
        
        # Activation and normalization
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(normalized_shape=in_channels)

    def forward(self, x):
        """
        Forward pass of the ConvNeXt block.

        Args:
            x (torch.Tensor): Input feature map of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature map after applying the block operations.
        """
        residual = x

        # Apply depthwise convolution
        x = self.depthwise(x)
        # Apply LayerNorm in channel-last format and permute back
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        # Apply pointwise convolutions and activation
        x = self.pointwise1(x)
        x = self.act(x)
        x = self.pointwise2(x)
        # Add residual connection
        x += residual

        return x


class ConvNeXtT(nn.Module):
    """
    ConvNeXt-T architecture for image classification. It consists of a stem convolution
    followed by four stages with repeated ConvNeXt blocks, downsampling between stages,
    and a final classification layer.

    Args:
        in_channels (int): Number of input channels. Default is 3 for RGB images.
        num_classes (int): Number of output classes for classification. Default is 1000 (ImageNet).
        channel_config (list): List containing the number of channels at each stage. 
                               Default is [96, 192, 384, 768].
        repeat_config (list): List containing the number of ConvNeXt blocks to repeat at each stage. 
                              Default is [3, 3, 9, 3].
    """
    def __init__(self, in_channels=3, num_classes=1000, channel_config=[96, 192, 384, 768], repeat_config=[3, 3, 9, 3]):
        super().__init__()

        # Stem convolution for early feature extraction
        self.stemconv = nn.Conv2d(in_channels=in_channels, out_channels=channel_config[0], kernel_size=4, stride=4)
        self.stemnorm = nn.LayerNorm(normalized_shape=channel_config[0])

        # Stage 2 with downsampling and repeated ConvNeXt blocks
        self.res2 = nn.Sequential(
            *[ConvNeXtBlock(in_channels=channel_config[0]) for i in range(repeat_config[0])]
        )
        self.downsample2norm = nn.LayerNorm(normalized_shape=channel_config[0])
        self.downsample2conv = nn.Conv2d(in_channels=channel_config[0], out_channels=channel_config[1], kernel_size=2, stride=2)

        # Stage 3
        self.res3 = nn.Sequential(
            *[ConvNeXtBlock(in_channels=channel_config[1]) for i in range(repeat_config[1])]
        )
        self.downsample3norm = nn.LayerNorm(normalized_shape=channel_config[1])
        self.downsample3conv = nn.Conv2d(in_channels=channel_config[1], out_channels=channel_config[2], kernel_size=2, stride=2)

        # Stage 4
        self.res4 = nn.Sequential(
            *[ConvNeXtBlock(in_channels=channel_config[2]) for i in range(repeat_config[2])]
        )
        self.downsample4norm = nn.LayerNorm(normalized_shape=channel_config[2])
        self.downsample4conv = nn.Conv2d(in_channels=channel_config[2], out_channels=channel_config[3], kernel_size=2, stride=2)

        # Stage 5
        self.res5 = nn.Sequential(
            *[ConvNeXtBlock(in_channels=channel_config[3]) for i in range(repeat_config[3])]
        )

        # Global average pooling and final classification layer
        self.globalavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  
        self.norm = nn.LayerNorm(normalized_shape=channel_config[3])
        self.fc = nn.Linear(in_features=channel_config[3], out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the ConvNeXtT model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits for classification.
        """
        # Stem block
        x = self.stemconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.stemnorm(x)
        x = x.permute(0, 3, 1, 2)

        # Stage 2
        x = self.res2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.downsample2norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample2conv(x)

        # Stage 3
        x = self.res3(x)
        x = x.permute(0, 2, 3, 1)
        x = self.downsample3norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample3conv(x)

        # Stage 4
        x = self.res4(x)
        x = x.permute(0, 2, 3, 1)
        x = self.downsample4norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.downsample4conv(x)

        # Stage 5
        x = self.res5(x)

        # Global average pooling and classification
        x = self.globalavgpool(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and move it to the GPU
    model = ConvNeXtT().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))
