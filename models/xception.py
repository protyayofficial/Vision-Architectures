import torch
import torch.nn as nn

class SeparableConv(nn.Module):
    """
    Implements a Separable Convolution, which combines a depthwise convolution 
    with a pointwise convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 1.
        stride (int, optional): Stride of the convolution. Default is 1.
        padding (int, optional): Zero-padding added to both sides of the input. Default is 0.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is False.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, dilation=1, bias=bias)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, dilation=1, groups=in_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x
    
class Block(nn.Module):
    """
    Implements a block consisting of a ReLU activation, SeparableConv, and Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        bias (bool): If True, adds a learnable bias to the output.
        pre_activation (bool, optional): If True, applies ReLU activation before the convolution. Default is True.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, pre_activation=True):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.separable = SeparableConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.pre_activation = pre_activation

    def forward(self, x):
        if self.pre_activation:
            x = self.relu(x)
        x = self.separable(x)
        x = self.bn(x)
        return x
    
class ConvBlock(nn.Module):
    """
    Implements a basic convolutional block with Conv2D, BatchNorm, and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Zero-padding added to both sides of the input.
        bias (bool): If True, adds a learnable bias to the output.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class EntryFlow(nn.Module):
    """
    Implements the Entry Flow of the Xception network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False)
        self.conv2 = ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0, bias=False)
        self.downsample1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.block1 = nn.Sequential(
            Block(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=False),
            Block(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=256)
        self.block2 = nn.Sequential(
            Block(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            Block(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.downsample3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.block3 = nn.Sequential(
            Block(in_channels=256, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            Block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        out1 = self.block1(x)
        residual1 = self.downsample1(x)
        residual1 = self.bn1(residual1)
        x = out1 + residual1
        out2 = self.block2(x)
        residual2 = self.downsample2(x)
        residual2 = self.bn2(residual2)
        x = out2 + residual2
        out3 = self.block3(x)
        residual3 = self.downsample3(x)
        residual3 = self.bn3(residual3)
        x = out3 + residual3
        return x
    
class MiddleFlow(nn.Module):
    """
    Implements the Middle Flow of the Xception network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            Block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            Block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            Block(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
        )

    def forward(self, x):
        out = self.block(x)
        x += out
        return x

class ExitFlow(nn.Module):
    """
    Implements the Exit Flow of the Xception network.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (typically the number of classes).
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample1 = nn.Conv2d(in_channels=in_channels, out_channels=1024, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.block1 = nn.Sequential(
            Block(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            Block(in_channels=in_channels, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            Block(in_channels=1024, out_channels=1536, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=False),
            nn.ReLU(inplace=True),
            Block(in_channels=1536, out_channels=2048, kernel_size=3, stride=1, padding=1, bias=False, pre_activation=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=2048, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        out1 = self.block1(x)
        residual1 = self.downsample1(x)
        residual1 = self.bn1(residual1)
        x = out1 + residual1
        x = self.block2(x)
        return x

class Xception(nn.Module):
    """
    Implements the full Xception network with Entry Flow, Middle Flow, and Exit Flow.

    Args:
        in_channels (int, optional): Number of input channels (e.g., 3 for RGB images). Default is 3.
        out_channels (int, optional): Number of output channels (typically the number of classes). Default is 1000 (ImageNet classes).
    """

    def __init__(self, in_channels=3, out_channels=1000):
        super().__init__()
        self.entry = EntryFlow(in_channels=in_channels, out_channels=728)
        self.middle = nn.Sequential(
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728),
            MiddleFlow(in_channels=728, out_channels=728)
        )
        self.exit = ExitFlow(in_channels=728, out_channels=out_channels)

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = torch.flatten(x, start_dim=1)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the Xception model and move it to the GPU
    model = Xception().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 299, 299)))