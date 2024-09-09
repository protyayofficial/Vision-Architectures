import torch
import torch.nn as nn

class ResidualUnit(nn.Module):
    """
    Implements a basic residual block with bottleneck architecture.
    This block follows the pattern of Conv-BN-ReLU and allows residual connections 
    for gradient flow.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the convolution. Default is 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # First batch norm and 1x1 convolution layer
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)

        # Second batch norm and 3x3 convolution layer
        self.bn2 = nn.BatchNorm2d(num_features=out_channels // 4)
        self.conv2 = nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels // 4, kernel_size=3, stride=stride, padding=1, bias=False)

        # Third batch norm and 1x1 convolution layer
        self.bn3 = nn.BatchNorm2d(num_features=out_channels // 4)
        self.conv3 = nn.Conv2d(in_channels=out_channels // 4, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Conv layer for the residual path when in_channels != out_channels or stride != 1
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

        self.act = nn.ReLU(inplace=True)  # Activation function

    def forward(self, x):
        """
        Forward pass through the ResidualUnit.

        Args:
            x: Input tensor.
        
        Returns:
            Tensor after passing through residual block.
        """
        residual = x

        # BatchNorm -> ReLU -> Conv sequence
        out = self.bn1(x)
        out_ = self.act(out)
        out = self.conv1(out_)
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        # Apply residual connection if dimensions don't match
        if (self.in_channels != self.out_channels) or (self.stride != 1):
            residual = self.conv4(out_)

        out += residual  # Add the skip connection
        return out
    
class AttentionStage1(nn.Module):
    """
    Implements the first stage of the attention mechanism using residual units.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size1 (tuple): Output size for the first upsampling layer.
        size2 (tuple): Output size for the second upsampling layer.
        size3 (tuple): Output size for the third upsampling layer.
    """
    def __init__(self, in_channels, out_channels, size1=(56, 56), size2=(28, 28), size3=(14, 14)):
        super().__init__()

        self.preprocessing = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

        # Trunk branch of the attention mechanism (feature processing)
        self.trunk = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
        )

        # Mask branch of the attention mechanism (soft attention)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.residual1 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.residual2 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax3 = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        )

        # Upsampling and combining soft attention outputs
        self.interpolate3 = nn.UpsamplingBilinear2d(size=size3)
        self.softmax4 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.interpolate2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax5 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.interpolate1 = nn.UpsamplingBilinear2d(size=size1)

        # Final softmax output
        self.softmax6 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.lastblock = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        """
        Forward pass through AttentionStage1.

        Args:
            x: Input tensor.
        
        Returns:
            Tensor processed by attention mechanism.
        """
        # Preprocessing residual unit
        x = self.preprocessing(x)

        # Trunk branch
        out_trunk = self.trunk(x)

        # Soft attention branch
        out_maxpool1 = self.maxpool1(x)
        out_softmax1 = self.softmax1(out_maxpool1)
        out_residual1 = self.residual1(out_softmax1)

        out_maxpool2 = self.maxpool2(out_softmax1)
        out_softmax2 = self.softmax2(out_maxpool2)
        out_residual2 = self.residual2(out_softmax2)

        out_maxpool3 = self.maxpool3(out_softmax2)
        out_softmax3 = self.softmax3(out_maxpool3)

        # Upsampling and adding intermediate soft attention outputs
        out_interpolate3 = self.interpolate3(out_softmax3) + out_softmax2
        out = out_interpolate3 + out_residual2

        out_softmax4 = self.softmax4(out)
        out_interpolate2 = self.interpolate2(out_softmax4) + out_softmax1
        out = out_interpolate2 + out_residual1

        out_softmax5 = self.softmax5(out)
        out_interpolate1 = self.interpolate1(out_softmax5) + out_trunk

        out_softmax6 = self.softmax6(out_interpolate1)

        # Final output after applying soft attention
        out = (1 + out_softmax6) * out_trunk

        out = self.lastblock(out)

        return out
    
class AttentionStage2(nn.Module):
    """
    Implements the second stage of the attention mechanism.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size1 (tuple): Output size for the first upsampling layer.
        size2 (tuple): Output size for the second upsampling layer.
    """
    def __init__(self, in_channels, out_channels, size1=(28, 28), size2=(14, 14)):
        super().__init__()

        self.preprocessing = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

        # Trunk branch of the attention mechanism (feature processing)
        self.trunk = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        )

        # Mask branch of the attention mechanism (soft attention)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.residual1 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax2 = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
        )

        # Upsampling and combining soft attention outputs
        self.interpolate2 = nn.UpsamplingBilinear2d(size=size2)
        self.softmax3 = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        self.interpolate1 = nn.UpsamplingBilinear2d(size=size1)
        
        # Final softmax output
        self.softmax4 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.lastblock = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        """
        Forward pass through AttentionStage2.

        Args:
            x: Input tensor.
        
        Returns:
            Tensor processed by attention mechanism.
        """
        # Preprocessing residual unit
        x = self.preprocessing(x)

        # Trunk branch
        out_trunk = self.trunk(x)

        # Soft attention branch
        out_maxpool1 = self.maxpool1(x)
        out_softmax1 = self.softmax1(out_maxpool1)
        out_residual1 = self.residual1(out_softmax1)

        out_maxpool2 = self.maxpool2(out_softmax1)
        out_softmax2 = self.softmax2(out_maxpool2)

        # Upsampling and adding intermediate soft attention outputs
        out_interpolate2 = self.interpolate2(out_softmax2) + out_softmax1
        out = out_interpolate2 + out_residual1

        out_softmax3 = self.softmax3(out)
        out_interpolate1 = self.interpolate1(out_softmax3) + out_trunk

        out_softmax4 = self.softmax4(out_interpolate1)

        # Final output after applying soft attention
        out = (1 + out_softmax4) * out_trunk

        out = self.lastblock(out)

        return out

class AttentionStage3(nn.Module):
    """
    Implements the third stage of the attention mechanism.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        size1 (tuple): Output size for the upsampling layer.
    """
    def __init__(self, in_channels, out_channels, size1=(14, 14)):
        super().__init__()

        self.preprocessing = ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        
        self.trunk = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.softmax1 = nn.Sequential(
            ResidualUnit(in_channels=in_channels, out_channels=out_channels),
            ResidualUnit(in_channels=in_channels, out_channels=out_channels)
        )

        self.interpolate1 = nn.UpsamplingBilinear2d(size=size1)

        self.softmax2 = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

        self.lastblock = ResidualUnit(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        """
        Forward pass through AttentionStage3.

        Args:
            x: Input tensor.
        
        Returns:
            Tensor processed by attention mechanism.
        """
        x = self.preprocessing(x)

        out_trunk = self.trunk(x)

        out_maxpool1 = self.maxpool1(x)
        out_softmax1 = self.softmax1(out_maxpool1)

        out_interpolate1 = self.interpolate1(out_softmax1) + out_trunk

        out_softmax2 = self.softmax2(out_interpolate1)
        out = (1 + out_softmax2) * out_trunk

        out = self.lastblock(out)

        return out
    
class ResidualAttentionNetwork92(nn.Module):
    """
    Residual Attention Network with 92 layers.
    
    Args:
        in_channels (int): Number of input channels. Default is 3 (RGB images).
        num_classes (int): Number of output classes for classification. Default is 1000 (ImageNet).
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.residualunit1 = ResidualUnit(in_channels=64, out_channels=256)

        self.attentionstage1 = AttentionStage1(in_channels=256, out_channels=256)

        self.residualunit2 = ResidualUnit(in_channels=256, out_channels=512, stride=2)

        self.attentionstage2_1 = AttentionStage2(in_channels=512, out_channels=512)
        self.attentionstage2_2 = AttentionStage2(in_channels=512, out_channels=512)

        self.residualunit3 = ResidualUnit(in_channels=512, out_channels=1024, stride=2)

        self.attentionstage3_1 = AttentionStage3(in_channels=1024, out_channels=1024)
        self.attentionstage3_2 = AttentionStage3(in_channels=1024, out_channels=1024)
        self.attentionstage3_3 = AttentionStage3(in_channels=1024, out_channels=1024)

        self.residualunit4 = ResidualUnit(in_channels=1024, out_channels=2048, stride=2)
        self.residualunit5 = ResidualUnit(in_channels=2048, out_channels=2048)
        self.residualunit6 = ResidualUnit(in_channels=2048, out_channels=2048)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_features=2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the Residual Attention Network.

        Args:
            x: Input tensor.
        
        Returns:
            Output tensor with class scores.
        """
        out = self.conv1(x)
        out = self.maxpool1(out)

        out = self.residualunit1(out)
        out = self.attentionstage1(out)
        out = self.residualunit2(out)
        out = self.attentionstage2_1(out)
        out = self.attentionstage2_2(out)
        out = self.residualunit3(out)
        out = self.attentionstage3_1(out)
        out = self.attentionstage3_2(out)
        out = self.attentionstage3_3(out)
        out = self.residualunit4(out)
        out = self.residualunit5(out)
        out = self.residualunit6(out)

        
        out = self.classifier(out)

        return out
       
if __name__ == "__main__":
    from torchsummary import summary

    model = ResidualAttentionNetwork92().to('cuda')

    print(summary(model, (3, 224, 224)))






    
