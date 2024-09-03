import torch
import torch.nn as nn

class DepthWiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        """
        Implements a depthwise separable convolution, which is a type of convolution that 
        reduces the computational cost compared to the standard convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
            padding (int): Padding added to all four sides of the input.
        """
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=padding, bias=False, groups=in_channels, dilation=1)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dilation=1)

        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the DepthWiseSeparableConv layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying depthwise separable convolution.
        """
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
class MobileNetV1(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        """
        Implements the MobileNetV1 architecture using depthwise separable convolutions. 
        MobileNet is designed to be lightweight and efficient, making it suitable 
        for mobile and embedded vision applications.

        Args:
            in_channels (int, optional): Number of input channels (e.g., 3 for RGB images). Default is 3.
            num_classes (int, optional): Number of output classes for the final classification. Default is 1000.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        
        self.relu = nn.ReLU()

        self.block1 = DepthWiseSeparableConv(in_channels=32, out_channels=64, stride=1, padding=1)

        self.block2 = DepthWiseSeparableConv(in_channels=64, out_channels=128, stride=2, padding=1)

        self.block3 = DepthWiseSeparableConv(in_channels=128, out_channels=128, stride=1, padding=1)

        self.block4 = DepthWiseSeparableConv(in_channels=128, out_channels=256, stride=2, padding=1)

        self.block5 = DepthWiseSeparableConv(in_channels=256, out_channels=256, stride=1, padding=1)

        self.block6 = DepthWiseSeparableConv(in_channels=256, out_channels=512, stride=2, padding=1)

        self.block7 = nn.Sequential(
            DepthWiseSeparableConv(in_channels=512, out_channels=512, stride=1, padding=1),
            DepthWiseSeparableConv(in_channels=512, out_channels=512, stride=1, padding=1),
            DepthWiseSeparableConv(in_channels=512, out_channels=512, stride=1, padding=1),
            DepthWiseSeparableConv(in_channels=512, out_channels=512, stride=1, padding=1),
            DepthWiseSeparableConv(in_channels=512, out_channels=512, stride=1, padding=1),
        )

        self.block8 = DepthWiseSeparableConv(in_channels=512, out_channels=1024, stride=2, padding=1)

        self.block9 = DepthWiseSeparableConv(in_channels=1024, out_channels=1024, stride=1, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the MobileNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the MobileNet architecture.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.avgpool(x)
        
        # Flatten the tensor before passing to the fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV1().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))