import torch
import torch.nn as nn

class BottleNeck(nn.Module):
    """
    A bottleneck block for the MobileNetV2 architecture, consisting of a sequence of convolutional layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the depthwise convolution.
        padding (int): Padding applied to the depthwise convolution.
        expansion_factor (int, optional): Expansion factor to increase the number of channels in the bottleneck. Default is 6.
    """
    def __init__(self, in_channels, out_channels, stride, padding, expansion_factor=6):
        super().__init__()

        # Pointwise convolution to expand the number of channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=expansion_factor * in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=expansion_factor * in_channels)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels=expansion_factor * in_channels, out_channels=expansion_factor * in_channels, kernel_size=3, stride=stride, padding=padding, bias=False, groups=expansion_factor * in_channels)
        self.bn2 = nn.BatchNorm2d(num_features=expansion_factor * in_channels)

        # Pointwise convolution to project back to the desired output channels
        self.linear = nn.Conv2d(in_channels=expansion_factor * in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        # Non-linearity used in the block
        self.relu = nn.ReLU6()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the bottleneck block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the bottleneck block.
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.depthwise(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.linear(out)
        out = self.bn3(out)

        if self.stride == 1 and self.in_channels == self.out_channels:
            out += x

        return out


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model architecture for efficient image classification.
    
    Args:
        in_channels (int, optional): Number of input channels. Default is 3 for RGB images.
        num_classes (int, optional): Number of output classes for classification. Default is 1000.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU6()

        # Bottleneck blocks
        self.block1 = BottleNeck(in_channels=32, out_channels=16, stride=1, padding=1, expansion_factor=1)

        self.block2 = nn.Sequential(
            BottleNeck(in_channels=16, out_channels=24, stride=2, padding=1),
            BottleNeck(in_channels=24, out_channels=24, stride=1, padding=1)
        )

        self.block3 = nn.Sequential(
            BottleNeck(in_channels=24, out_channels=32, stride=2, padding=1),
            BottleNeck(in_channels=32, out_channels=32, stride=1, padding=1),
            BottleNeck(in_channels=32, out_channels=32, stride=1, padding=1),
        )

        self.block4 = nn.Sequential(
            BottleNeck(in_channels=32, out_channels=64, stride=2, padding=1),
            BottleNeck(in_channels=64, out_channels=64, stride=1, padding=1),
            BottleNeck(in_channels=64, out_channels=64, stride=1, padding=1),
            BottleNeck(in_channels=64, out_channels=64, stride=1, padding=1),            
        )

        self.block5 = nn.Sequential(
            BottleNeck(in_channels=64, out_channels=96, stride=1, padding=1),
            BottleNeck(in_channels=96, out_channels=96, stride=1, padding=1),
            BottleNeck(in_channels=96, out_channels=96, stride=1, padding=1),
        )

        self.block6 = nn.Sequential(
            BottleNeck(in_channels=96, out_channels=160, stride=2, padding=1),
            BottleNeck(in_channels=160, out_channels=160, stride=1, padding=1),
            BottleNeck(in_channels=160, out_channels=160, stride=1, padding=1),
        )

        self.block7 = BottleNeck(in_channels=160, out_channels=320, stride=1, padding=1)

        # Final convolution and pooling layers
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=1280)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=1280, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass of the MobileNetV2 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with predictions for each class.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
    
        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV2().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))
