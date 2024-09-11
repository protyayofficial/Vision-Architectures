import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_channels // reduction_ratio, out_channels=in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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
    def __init__(self, kernel_size):
        super().__init__()

        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out_max = self.maxpool(x)
        out_avg = self.avgpool(x)

        out = torch.cat([out_max, out_avg], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return out

class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size, reduction_ratio=16):
        super().__init__()

        self.channel_attention = ChannelAttention(in_channels=in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out_ca = self.channel_attention(x)
        out_ca = out_ca * x

        out_sa = self.spatial_attention(out_ca)
        out_sa = out_sa * out_ca

        return out_sa

class Bottleneck(nn.Module):
    """
    A basic Bottleneck block for ResNet. It consists of three convolutional layers with a possible downsampling 
    (via a stride of 2) and a skip connection (identity mapping) that bypasses these layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolutional layer. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()

        self.expansion = 4

        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels // self.expansion, kernel_size=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels // self.expansion)
        self.relu = nn.ReLU()

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=out_channels // self.expansion, out_channels=out_channels // self.expansion, kernel_size=3, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels // self.expansion)

        # Third Convolutional Layer
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
        # Save the input (identity) for the skip connection
        identity = x

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

        out = self.cbam(out)

        # If there's a downsample layer, apply it to the identity
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  # Apply ReLU after the addition according to He et al.

        return out
    

class CBAMResNet50(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(CBAMResNet50, self).__init__()

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
                 
        layers = []
    
        # First block may downsample and/or change the number of channels
        layers.append(Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride))

        for _ in range(1, blocks):
            layers.append(Bottleneck(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
       
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
    model = CBAMResNet50().to('cuda')
    # Print a summary of the model
    print(summary(model, (3, 224, 224)))