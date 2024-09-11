import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A basic convolutional block consisting of a convolutional layer followed by an optional activation layer (ReLU).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        padding (int): Padding applied to the input.
        stride (int): Stride of the convolutional kernel.
        use_act (bool): Whether to apply the activation function. Default is True.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, use_act=True):
        super().__init__()
        self.use_act = use_act

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution (and optional activation).
        """
        out = self.conv(x)
        if self.use_act:
            out = self.act(out)
        return out

class FireModule(nn.Module):
    """
    The Fire Module is a key building block of SqueezeNet. It consists of a squeeze layer (1x1 convolutions) 
    followed by an expand layer (both 1x1 and 3x3 convolutions).

    Args:
        in_channels (int): Number of input channels.
        s1x1 (int): Number of filters for the squeeze layer (1x1 convolution).
        e1x1 (int): Number of filters for the expand 1x1 convolution layer.
        e3x3 (int): Number of filters for the expand 3x3 convolution layer.
    """
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super().__init__()

        self.squeeze1x1 = ConvBlock(in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1, padding=0)
        self.expand1x1 = ConvBlock(in_channels=s1x1, out_channels=e1x1, kernel_size=1, stride=1, padding=0, use_act=False)
        self.expand3x3 = ConvBlock(in_channels=s1x1, out_channels=e3x3, kernel_size=3, stride=1, padding=1, use_act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the Fire module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenation of 1x1 and 3x3 expanded outputs.
        """
        out = self.squeeze1x1(x)
        out1 = self.expand1x1(out)
        out2 = self.expand3x3(out)
        out = torch.cat([out1, out2], dim=1)
        return out

class SqueezeNet(nn.Module):
    """
    The SqueezeNet architecture that uses Fire modules to reduce the number of parameters while maintaining performance.
    
    Args:
        in_channels (int): Number of input channels (default is 3 for RGB images).
        num_classes (int): Number of output classes for classification.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=96, kernel_size=7, stride=2, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)
        self.maxpool8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)
        self.conv10 = ConvBlock(in_channels=512, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass through the SqueezeNet model.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        out = self.conv1(x)
        out = self.maxpool1(out)
        out = self.fire2(out)
        out = self.fire3(out)
        out = self.fire4(out)
        out = self.maxpool4(out)
        out = self.fire5(out)
        out = self.fire6(out)
        out = self.fire7(out)
        out = self.fire8(out)
        out = self.maxpool8(out)
        out = self.fire9(out)
        out = self.conv10(out)
        out = self.dropout(out)
        out = self.avgpool(out)

        return out

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the SqueezeNet model and move it to GPU
    model = SqueezeNet().to('cuda')

    # Print a summary of the model for input size (3, 224, 224)
    print(summary(model, (3, 224, 224)))
