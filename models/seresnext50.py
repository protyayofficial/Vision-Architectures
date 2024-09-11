import torch
import torch.nn as nn

class SqueezeExcitation(nn.Module):
    """
    Implements the Squeeze-and-Excitation (SE) block.
    It recalibrates the channel-wise feature responses by modeling the interdependencies between channels.
    """
    def __init__(self, in_channels, reduction_factor):
        """
        Initializes the SqueezeExcitation block.

        Args:
            in_channels (int): Number of input channels.
            reduction_factor (int): Reduction factor for the squeeze operation.
        """
        super().__init__()

        # Squeeze operation reduces the number of channels
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_factor, 
                                 kernel_size=1, stride=1, padding=0, bias=False)
        # Excitation operation restores the number of channels
        self.excitation = nn.Conv2d(in_channels=in_channels // reduction_factor, out_channels=in_channels, 
                                    kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        # Global Average Pooling to reduce the spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x):
        """
        Forward pass through the SE block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).
        
        Returns:
            Tensor: Output tensor recalibrated by channel-wise dependencies.
        """
        # Global Average Pooling
        out = self.avgpool(x)
        # Squeeze operation
        out = self.squeeze(out)
        out = self.relu(out)
        # Excitation operation
        out = self.excitation(out)
        out = self.sigmoid(out)

        # Scale the input feature map by the recalibrated activations
        return x * out

class ResNeXtBlock(nn.Module):
    """
    Implements a ResNeXt block with a Squeeze-and-Excitation (SE) module.
    The block uses grouped convolutions to reduce computational complexity.
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32, reduction_factor=16):
        """
        Initializes the ResNeXtBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first convolutional layer.
            cardinality (int): Number of groups for the second convolutional layer.
            reduction_factor (int): Reduction factor for the SE block.
        """
        super(ResNeXtBlock, self).__init__()

        self.expansion = 2

        # First Convolutional Layer (1x1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // self.expansion, 
                               kernel_size=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels // self.expansion)
        self.relu = nn.ReLU()

        # Second Convolutional Layer (Grouped 3x3)
        self.conv2 = nn.Conv2d(in_channels=out_channels // self.expansion, 
                               out_channels=out_channels // self.expansion, kernel_size=3, 
                               padding=1, bias=False, groups=cardinality)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels // self.expansion)

        # Third Convolutional Layer (1x1)
        self.conv3 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels, 
                               kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(num_features=out_channels)

        # Squeeze-and-Excitation module
        self.se = SqueezeExcitation(in_channels=out_channels, reduction_factor=reduction_factor)

        # Downsampling Layer if necessary
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        """
        Forward pass through the ResNeXt block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).
        
        Returns:
            Tensor: Output tensor after passing through the block.
        """
        identity = x  # Save the input for skip connection

        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Second conv -> BN -> ReLU
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        # Third conv -> BN and SqueezeExcitation
        out = self.conv3(out)
        out = self.batchnorm3(out)
        out = self.se(out)

        # If downsample is necessary, apply it to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        # Add skip connection
        out += identity
        out = self.relu(out)

        return out


class SEResNeXt50(nn.Module):
    """
    SEResNeXt50 architecture with Squeeze-and-Excitation modules.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        """
        Initializes the SEResNeXt50 model.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
        """
        super().__init__()

        # Initial Convolution + BatchNorm + ReLU + MaxPool layer
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Four stages of ResNeXt blocks with grouped convolutions and SE modules
        self.stage1 = self._make_stage(in_channels=64, out_channels=256, blocks=3)
        self.stage2 = self._make_stage(in_channels=256, out_channels=512, blocks=4, stride=2)
        self.stage3 = self._make_stage(in_channels=512, out_channels=1024, blocks=6, stride=2)
        self.stage4 = self._make_stage(in_channels=1024, out_channels=2048, blocks=3, stride=2)

        # Final pooling and fully connected layer
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """
        Creates a stage composed of several ResNeXt blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of ResNeXt blocks in the stage.
            stride (int): Stride for the first block in the stage.

        Returns:
            nn.Sequential: A sequential container of ResNeXt blocks.
        """
        layers = []

        # Add the first block with the specified stride
        layers.append(ResNeXtBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))

        # Add remaining blocks with stride=1
        for _ in range(1, blocks):
            layers.append(ResNeXtBlock(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the SEResNeXt50 model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, H, W).
        
        Returns:
            Tensor: Output tensor after classification.
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
    model = SEResNeXt50().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))
