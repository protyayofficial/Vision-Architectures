import torch
import torch.nn as nn

class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, expansion_factor):
        super().__init__()

        # Pointwise convolution to expand the number of channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=expansion_factor * in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=expansion_factor * in_channels)

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels=expansion_factor * in_channels, out_channels=expansion_factor * in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=expansion_factor * in_channels)
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
    
class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False, dilation=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU6(),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1, dilation=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
        )

    def forward(self, x):
        return self.block(x)


class MNASNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU6()

        # Bottleneck blocks
        self.sepconv = SepConv(in_channels=32, out_channels=16)

        self.block1 = nn.Sequential(
            InvertedResidual(in_channels=16, out_channels=24, kernel_size=3, stride=2, padding=1, expansion_factor=3),
            InvertedResidual(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, expansion_factor=3),
            InvertedResidual(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1, expansion_factor=3),
        )

        self.block2 = nn.Sequential(
            InvertedResidual(in_channels=24, out_channels=40, kernel_size=5, stride=2, padding=2, expansion_factor=3),
            InvertedResidual(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2, expansion_factor=3),
            InvertedResidual(in_channels=40, out_channels=40, kernel_size=5, stride=1, padding=2, expansion_factor=3),
        )

        self.block3 = nn.Sequential(
            InvertedResidual(in_channels=40, out_channels=80, kernel_size=5, stride=2, padding=2, expansion_factor=6),
            InvertedResidual(in_channels=80, out_channels=80, kernel_size=5, stride=1, padding=2, expansion_factor=6),
            InvertedResidual(in_channels=80, out_channels=80, kernel_size=5, stride=1, padding=2, expansion_factor=6),     
        )

        self.block4 = nn.Sequential(
            InvertedResidual(in_channels=80, out_channels=96, kernel_size=3, stride=1, padding=1, expansion_factor=6),
            InvertedResidual(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, expansion_factor=6),
        )

        self.block5 = nn.Sequential(
            InvertedResidual(in_channels=96, out_channels=192, kernel_size=5, stride=2, padding=2, expansion_factor=6),
            InvertedResidual(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2, expansion_factor=6),
            InvertedResidual(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2, expansion_factor=6),
            InvertedResidual(in_channels=192, out_channels=192, kernel_size=5, stride=1, padding=2, expansion_factor=6),
        )

        self.block6 = InvertedResidual(in_channels=192, out_channels=320, kernel_size=3, stride=1, padding=1, expansion_factor=6)

        # Final convolution and pooling layers
        self.conv2 = nn.Conv2d(in_channels=320, out_channels=1280, kernel_size=1, stride=1, padding=0, bias=False)
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

        x = self.sepconv(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
    
        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    model = MNASNet().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))
