import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block that performs feature recalibration by 
    emphasizing informative features and suppressing less useful ones.

    Args:
        in_channels (int): Number of input channels.
        reduction_factor (int): Factor by which the number of channels is reduced during the squeeze operation.
    """
    def __init__(self, in_channels, reduction_factor):
        super().__init__()

        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_factor, kernel_size=1, stride=1, padding=0, bias=False)
        self.excitation = nn.Conv2d(in_channels=in_channels // reduction_factor, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x):
        """
        Forward pass for the Squeeze-and-Excitation block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Recalibrated feature map.
        """
        out = self.avgpool(x)
        out = self.squeeze(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)

        return x * out


class Bottleneck(nn.Module):
    """
    Bottleneck block for the SE-ResNet architecture, with Squeeze-and-Excitation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolution layer. Default: 1.
        reduction_factor (int, optional): Reduction factor for the SE block. Default: 16.
    """
    def __init__(self, in_channels, out_channels, stride=1, reduction_factor=16):
        super(Bottleneck, self).__init__()

        self.expansion = 4

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // self.expansion, kernel_size=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels // self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels // self.expansion, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels // self.expansion)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels)

        self.downsample = None

        # Squeeze-and-Excitation block
        self.se = SqueezeExcitation(in_channels=out_channels, reduction_factor=reduction_factor)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        """
        Forward pass for the bottleneck block.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Output feature map after applying the bottleneck transformation.
        """
        identity = x  # Save the input for the skip connection

        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Second conv -> BN -> ReLU
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        # Third conv -> BN and SE block
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.se(out)

        # Apply downsample if needed
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Skip connection
        out = self.relu(out)

        return out
    

class SEResNet50(nn.Module):
    """
    SE-ResNet50 model with Squeeze-and-Excitation blocks.

    Args:
        in_channels (int, optional): Number of input channels. Default: 3.
        num_classes (int, optional): Number of output classes for the final fully connected layer. Default: 1000.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super(SEResNet50, self).__init__()

        # Initial convolution block
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Four stages of residual blocks
        self.stage1 = self._make_stage(in_channels=64, out_channels=256, blocks=3)
        self.stage2 = self._make_stage(in_channels=256, out_channels=512, blocks=4, stride=2)
        self.stage3 = self._make_stage(in_channels=512, out_channels=1024, blocks=6, stride=2)
        self.stage4 = self._make_stage(in_channels=1024, out_channels=2048, blocks=3, stride=2)

        # Final pooling and fully connected layer
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a stage composed of multiple bottleneck blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of bottleneck blocks in the stage.
            stride (int, optional): Stride for the first block. Default: 1.

        Returns:
            nn.Sequential: A sequential container of bottleneck blocks.
        """
        layers = []

        # First block may downsample and/or change the number of channels
        layers.append(Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride))

        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the SE-ResNet50 model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output logits from the model.
        """
        x = self.conv_init(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.adaptiveavgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc(x)

        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and move it to the GPU
    model = SEResNet50().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))
