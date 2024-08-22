import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                              kernel_size=kernel_size, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.batchnorm(out)
        out = self.relu(out)

        out = self.conv(out)
        out = self.batchnorm(out)
        # out = self.relu(out)

        return out


class PoolingResidual(nn.Module):
    """
    A pooling residual block with two convolutional layers and a downsampling step.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride for the convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                      kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.ReLU(),   
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling block with a single convolutional layer.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            # nn.ReLU()
        )

    def forward(self, x):
        x = self.block(x)
        return x


class ResNet18(nn.Module):
    """
    A PyTorch implementation of the ResNet18 architecture.
    
    Args:
        in_channels (int, optional): Number of input channels. Default is 3.
        num_classes (int, optional): Number of output classes. Default is 1000.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        self.relu = nn.ReLU()

        # First residual block (64 channels)
        self.res_64_1 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)
        self.res_64_2 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3)
        
        # Pooling and upsampling for transition to 128 channels
        self.pool_res_128 = PoolingResidual(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.upsample_128 = Upsample(in_channels=64, out_channels=128)

        # Second residual block (128 channels)
        self.res_128_1 = ResidualBlock(in_channels=128, out_channels=128, kernel_size=3)

        # Pooling and upsampling for transition to 256 channels
        self.pool_res_256 = PoolingResidual(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.upsample_256 = Upsample(in_channels=128, out_channels=256)

        # Third residual block (256 channels)
        self.res_256_1 = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3)

        # Pooling and upsampling for transition to 512 channels
        self.pool_res_512 = PoolingResidual(in_channels=256, out_channels=512, kernel_size=3, stride=2)
        self.upsample_512 = Upsample(in_channels=256, out_channels=512)

        # Fourth residual block (512 channels)
        self.res_512_1 = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3)

        # Final pooling and fully connected layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=7)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # Initial convolution and max pooling
        x = self.conv_init(x)
        x = self.maxpool(x)

        # First residual block
        out = self.res_64_1(x)
        x = out + x
        x = self.relu(x)

        out = self.res_64_2(x)
        x = out + x
        x = self.relu(x)

        # Transition to 128 channels
        upsampledoutput = self.upsample_128(x)
        x = self.pool_res_128(x)
        x = upsampledoutput + x
        x = self.relu(x)

        out = self.res_128_1(x)
        x = out + x
        x = self.relu(x)

        # Transition to 256 channels
        upsampledoutput = self.upsample_256(x)
        x = self.pool_res_256(x)
        x = upsampledoutput + x
        x = self.relu(x)

        out = self.res_256_1(x)
        x = out + x
        x = self.relu(x)

        # Transition to 512 channels
        upsampledoutput = self.upsample_512(x)
        x = self.pool_res_512(x)
        x = upsampledoutput + x
        x = self.relu(x)

        out = self.res_512_1(x)
        x = out + x
        x = self.relu(x)

        # Final pooling and classification layer
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the ResNet18 model and move it to the GPU
    model = ResNet18().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))
