import torch
import torch.nn as nn

class TransitionLayer(nn.Module):
    """
    A transition layer in DenseNet that reduces the number of feature maps 
    and spatial dimensions.

    Args:
        in_channels (int): Number of input channels.
        compression_factor (float): Factor to reduce the number of channels.
    """
    def __init__(self, in_channels, compression_factor):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels * compression_factor), kernel_size=1, padding=0, bias=False)
        )

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass for the transition layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor with reduced spatial dimensions and number of channels.
        """
        x = self.block(x)
        x = self.avgpool(x)
        return x
    
class Bottleneck(nn.Module):
    """
    A bottleneck layer in DenseNet that increases the depth of the network.

    Args:
        in_channels (int): Number of input channels.
        growth_rate (int): Growth rate for increasing the number of channels.
    """
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.intermediate_planes = 4 * growth_rate

        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=self.intermediate_planes, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=self.intermediate_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.intermediate_planes, out_channels=growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        """
        Forward pass for the bottleneck layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Concatenated tensor of input and output along the channel dimension.
        """
        out = self.block(x)
        return torch.cat([x, out], dim=1)
    
class DenseBlock(nn.Module):
    """
    A dense block in DenseNet that consists of multiple bottleneck layers.

    Args:
        in_channels (int): Number of input channels.
        num_layers (int): Number of bottleneck layers in the dense block.
        growth_rate (int): Growth rate for increasing the number of channels.
    """
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.block = self._make_layers(in_channels, num_layers, growth_rate)

    def _make_layers(self, in_channels, num_layers, growth_rate):
        """
        Creates the bottleneck layers in the dense block.

        Args:
            in_channels (int): Number of input channels.
            num_layers (int): Number of bottleneck layers in the dense block.
            growth_rate (int): Growth rate for increasing the number of channels.

        Returns:
            nn.Sequential: Sequential container of bottleneck layers.
        """
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels=in_channels + growth_rate * i, growth_rate=growth_rate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for the dense block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the dense block.
        """
        x = self.block(x)
        return x
    
class DenseNet121(nn.Module):
    """
    Implementation of the DenseNet121 architecture.

    Args:
        in_channels (int, optional): Number of input channels. Default is 3.
        num_classes (int, optional): Number of output classes. Default is 1000.
        growth_rate (int, optional): Growth rate for increasing the number of channels. Default is 32.
        compression_factor (float, optional): Factor to reduce the number of channels in transition layers. Default is 0.5.
    """
    def __init__(self, in_channels=3, num_classes=1000, growth_rate=32, compression_factor=0.5):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        input_features = 2 * growth_rate 
        self.denseblock1 = DenseBlock(in_channels=input_features, num_layers=6, growth_rate=growth_rate)
        input_features = input_features + growth_rate * 6
        self.transitionblock1 = TransitionLayer(in_channels=input_features, compression_factor=compression_factor)
        input_features = int(input_features * compression_factor)

        self.denseblock2 = DenseBlock(in_channels=input_features, num_layers=12, growth_rate=growth_rate)
        input_features = input_features + growth_rate * 12
        self.transitionblock2 = TransitionLayer(in_channels=input_features, compression_factor=compression_factor)
        input_features = int(input_features * compression_factor)

        self.denseblock3 = DenseBlock(in_channels=input_features, num_layers=24, growth_rate=growth_rate)
        input_features = input_features + growth_rate * 24
        self.transitionblock3 = TransitionLayer(in_channels=input_features, compression_factor=compression_factor)
        input_features = int(input_features * compression_factor)

        self.denseblock4 = DenseBlock(in_channels=input_features, num_layers=16, growth_rate=growth_rate)
        input_features = input_features + growth_rate * 16

        self.batchnorm4 = nn.BatchNorm2d(num_features=input_features)
        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=input_features, out_features=num_classes, bias=False)

    def forward(self, x):
        """
        Forward pass for the DenseNet121 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, num_classes).
        """
        x = self.conv_init(x)

        x = self.denseblock1(x)
        x = self.transitionblock1(x)

        x = self.denseblock2(x)
        x = self.transitionblock2(x)

        x = self.denseblock3(x)
        x = self.transitionblock3(x)

        x = self.denseblock4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the model and print a summary
    model = DenseNet121()
    print(summary(model, (3, 224, 224)))

