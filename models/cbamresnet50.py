import torch
import torch.nn as nn

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Channel Attention module used to emphasize important features across channels.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int): Factor to reduce the number of channels for the attention mechanism.

    Methods:
        forward(x): Forward pass of the channel attention mechanism.
    """
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the channel attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying channel attention.
        """
        out_max = self.maxpool(x)
        out_avg = self.avgpool(x)

        out_max = self.conv1(out_max)
        out_max = self.relu(out_max)
        out_max = self.conv2(out_max)

        out_avg = self.conv1(out_avg)
        out_avg = self.relu(out_avg)
        out_avg = self.conv2(out_avg)

        out = out_avg + out_max
        out = self.sigmoid(out)

        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention module to focus on important spatial features.

    Args:
        kernel_size (int): Kernel size for the convolutional layer in spatial attention.

    Methods:
        forward(x): Forward pass of the spatial attention mechanism.
    """
    def __init__(self, kernel_size):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the spatial attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying spatial attention.
        """
        out_max = self.maxpool(x)
        out_avg = self.avgpool(x)

        out = torch.cat([out_max, out_avg], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) that combines both channel and spatial attention.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int): Kernel size for the spatial attention mechanism.
        reduction_ratio (int, optional): Reduction ratio for the channel attention. Default is 16.

    Methods:
        forward(x): Forward pass applying both channel and spatial attention.
    """
    def __init__(self, in_channels, kernel_size, reduction_ratio=16):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        """
        Forward pass for CBAM combining channel and spatial attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying CBAM attention.
        """
        out_ca = self.channel_attention(x)
        out_ca = out_ca * x

        out_sa = self.spatial_attention(out_ca)
        out_sa = out_sa * out_ca

        return out_sa


class Bottleneck(nn.Module):
    """
    A bottleneck block for ResNet, which includes CBAM.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Default is 1.

    Methods:
        forward(x): Forward pass of the bottleneck block.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.expansion = 4

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // self.expansion, kernel_size=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels // self.expansion)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels // self.expansion, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels // self.expansion)

        self.conv3 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels)

        self.cbam = CBAM(in_channels=out_channels, kernel_size=7)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        """
        Forward pass for the bottleneck block with CBAM attention.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the bottleneck block.
        """
        identity = x

        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.batchnorm3(out)

        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CBAMResNet50(nn.Module):
    """
    A ResNet-50 model with CBAM attention.

    Args:
        in_channels (int, optional): Number of input channels. Default is 3.
        num_classes (int, optional): Number of output classes. Default is 1000.

    Methods:
        forward(x): Forward pass for the entire CBAM-ResNet50 model.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super(CBAMResNet50, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = self._make_stage(in_channels=64, out_channels=256, blocks=3)
        self.stage2 = self._make_stage(in_channels=256, out_channels=512, blocks=4, stride=2)
        self.stage3 = self._make_stage(in_channels=512, out_channels=1024, blocks=6, stride=2)
        self.stage4 = self._make_stage(in_channels=1024, out_channels=2048, blocks=3, stride=2)

        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """
        Create a stage consisting of multiple Bottleneck blocks.

        Args:
            in_channels (int): Number of input channels for the stage.
            out_channels (int): Number of output channels for the stage.
            blocks (int): Number of Bottleneck blocks in the stage.
            stride (int, optional): Stride for the first Bottleneck block. Default is 1.

        Returns:
            nn.Sequential: A stage composed of Bottleneck blocks.
        """
        layers = []
        layers.append(Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride))

        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the CBAM-ResNet50 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits after passing through the network.
        """
        out = self.conv_init(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.adaptiveavgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
    
if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and move it to the GPU
    model = CBAMResNet50().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))