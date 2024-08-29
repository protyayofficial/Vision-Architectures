import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic residual block for ResNet. It consists of two convolutional layers with a possible downsampling 
    (via a stride of 2) and a skip connection (identity mapping) that bypasses these layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size remains same for all convolutional layers except when downsampling happens. Defaults to 3 as per He et al.
        stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        # Save the input (identity) for the skip connection
        identity = x

        # First conv -> BN -> ReLU
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        # Second conv -> BN
        out = self.conv2(out)
        out = self.batchnorm2(out)

        # If there's a downsample layer, apply it to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # Apply ReLU after the addition according to He et al.

        return out
    

class ResNet18(nn.Module):
    """
    ResNet18 model, consisting of an initial convolutional block followed by 4 stages of residual blocks.
    The number of channels doubles at each stage, and downsampling is performed between stages.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 3 (for RGB images).
        num_classes (int, optional): Number of output classes. Defaults to 1000 (for ImageNet).
    """

    def __init__(self, in_channels=3, num_classes=1000):
        super(ResNet18, self).__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Four stages of residual blocks
        self.stage1 = self._make_stage(in_channels=64, out_channels=64, blocks=2)
        self.stage2 = self._make_stage(in_channels=64, out_channels=128, blocks=2, stride=2)
        self.stage3 = self._make_stage(in_channels=128, out_channels=256, blocks=2, stride=2)
        self.stage4 = self._make_stage(in_channels=256, out_channels=512, blocks=2, stride=2)

        # Final pooling and fully connected layer
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride=1):
        """
        Helper function to create a layer with multiple residual blocks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of residual blocks in this layer.
            stride (int, optional): Stride for the first block. Defaults to 1.

        Returns:
            nn.Sequential: A sequential container of residual blocks.
        """
         
        layers = []
    
        # First block may downsample and/or change the number of channels
        layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels, stride=stride))

        for _ in range(1, blocks):
            layers.append(ResidualBlock(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Defines the forward pass for the ResNet18 model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output logits with shape (batch_size, num_classes).
        """

        x = self.conv_init(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.adaptiveavgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and move it to the GPU
    model = ResNet18().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))