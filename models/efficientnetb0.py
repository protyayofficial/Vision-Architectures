import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth
import math

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a Conv2d layer followed by BatchNorm and SiLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
            nn.BatchNorm2d(num_features=out_channels),
            nn.SiLU(inplace=True)  # Swish activation
        )

    def forward(self, x):
        """
        Forward pass through the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the ConvBlock.
        """
        return self.block(x)

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block to perform channel-wise feature recalibration.
    """
    def __init__(self, in_channels, reduced_dims):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=in_channels, out_channels=reduced_dims, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=reduced_dims, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the Squeeze-and-Excitation block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Scaled tensor after applying the squeeze-and-excitation block.
        """
        scale = self.se(x)
        return x * scale

class MBConv(nn.Module):
    """
    Mobile Inverted Residual Block (MBConv) with depthwise separable convolution, squeeze-and-excitation, and optional expansion.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, reduction_factor=4):
        super().__init__()

        intermediate_channels = in_channels * expansion_factor
        reduced_dims = max(1, in_channels // reduction_factor)

        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.expand = (expansion_factor != 1)

        if self.expand:
            self.expansion = ConvBlock(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=1, stride=1, padding=0)

        self.block = nn.Sequential(
            ConvBlock(in_channels=intermediate_channels, out_channels=intermediate_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=intermediate_channels),
            SqueezeExcitation(in_channels=intermediate_channels, reduced_dims=reduced_dims),
            nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )

    def forward(self, x):
        """
        Forward pass through the MBConv block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the MBConv block.
        """
        residual = x
        if self.expand:
            x = self.expansion(x)
        x = self.block(x)
        if self.use_residual:
            x += residual
        return x

class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 model with variable depth and width scaling.
    """
    def __init__(self, in_channels=3, phi=0, alpha=1.2, beta=1.1, num_classes=1000, dropout_rate=0.2):
        super().__init__()

        self.depth_coefficient = beta ** phi

        # [expansion factor, output channels, repeats, stride, kernel_size]
        stages = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        self.stage1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)

        input_channels = 32
        features = []

        for _, (expansion_factor, output_channels, num_repeats, stride, kernel_size) in enumerate(stages):
            num_repeats = int(self.scale_depth(num_repeats))

            for i in range(num_repeats):
                block_stride = stride if i == 0 else 1

                features.append(MBConv(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size, stride=block_stride, expansion_factor=expansion_factor))
                input_channels = output_channels

        self.features = nn.Sequential(*features)

        self.final_block = nn.Sequential(
            ConvBlock(in_channels=input_channels, out_channels=1280, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=1280, out_features=num_classes)
        )


    def scale_depth(self, depth):
        """
        Scale the depth of the network based on the depth coefficient.
        
        Args:
            depth (int): Number of repetitions to scale.
        
        Returns:
            int: Scaled number of repetitions.
        """
        return math.ceil(depth * self.depth_coefficient)

    def forward(self, x):
        """
        Forward pass through the EfficientNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the EfficientNet model.
        """
        x = self.stage1(x)
        x = self.features(x)
        x = self.final_block(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = EfficientNetB0().to("cuda")
    summary(model, (3, 224, 224))
