import torch
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    """
    A ResNeXt block that contains three convolutional layers and a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels after the final convolution.
        stride (int): Stride for the first convolution. Defaults to 1.
        cardinality (int): Number of groups for the grouped convolution in the second layer. Defaults to 32.
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(ResNeXtBlock, self).__init__()

        self.expansion = 2

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // self.expansion, kernel_size=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels // self.expansion)
        self.relu = nn.ReLU()

        # Second Convolutional Layer (Grouped Convolution)
        self.conv2 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels // self.expansion, kernel_size=3, padding=1, bias=False, groups=cardinality)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels // self.expansion)

        # Third Convolutional Layer
        self.conv3 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels)

        # Downsampling Layer (for matching dimensions in the skip connection)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        """
        Forward pass of the ResNeXt block.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        
        Returns:
            Tensor: Output tensor after passing through the ResNeXt block.
        """
        identity = x  # Save the input (identity) for the skip connection

        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Second conv -> BN -> ReLU
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        # Third conv -> BN
        out = self.conv3(out)
        out = self.batchnorm3(out)

        # If there's a downsample layer, apply it to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Add skip connection
        out = self.relu(out)  # Apply ReLU after the addition

        return out


class ResNeXt50(nn.Module):
    """
    ResNeXt-50 model based on grouped convolutions and bottleneck ResNeXt blocks.

    Args:
        in_channels (int): Number of input channels for the initial convolution layer. Defaults to 3 (RGB image).
        num_classes (int): Number of output classes for the final classification layer. Defaults to 1000.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        # Initial Convolution + BatchNorm + ReLU + MaxPool layer
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Four stages of residual blocks with grouped convolutions
        self.stage1 = self._make_stage(in_channels=64, out_channels=256, blocks=3)
        self.stage2 = self._make_stage(in_channels=256, out_channels=512, blocks=4, stride=2)
        self.stage3 = self._make_stage(in_channels=512, out_channels=1024, blocks=6, stride=2)
        self.stage4 = self._make_stage(in_channels=1024, out_channels=2048, blocks=3, stride=2)

        # Final pooling and fully connected layer
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a stage of ResNeXt blocks.
        
        Args:
            in_channels (int): Number of input channels for the first block.
            out_channels (int): Number of output channels after the blocks.
            blocks (int): Number of ResNeXt blocks in the stage.
            stride (int): Stride for the first block. Defaults to 1.
        
        Returns:
            Sequential: A sequence of ResNeXt blocks forming the stage.
        """
        layers = []

        # Add the first block with the specified stride
        layers.append(ResNeXtBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))

        # Add the remaining blocks with stride=1
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the ResNeXt-50 model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output tensor with logits for each class.
        """
        x = self.conv_init(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.adaptiveavgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and move it to the GPU
    model = ResNeXt50().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))
