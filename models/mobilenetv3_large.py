import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    """
    Hard sigmoid activation function.
    """
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU6()

    def forward(self, x):
        """
        Forward pass of the hard sigmoid activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying hard sigmoid.
        """
        return self.act(x + 3) / 6
    
class h_swish(nn.Module):
    """
    Hard Swish activation function.
    """
    def __init__(self):
        super().__init__()
        self.h_sigmoid = h_sigmoid()
    
    def forward(self, x):
        """
        Forward pass of the hard swish activation function.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying hard swish.
        """
        return x * self.h_sigmoid(x)
    
class SE(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=int(in_channels // reduction), kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=int(in_channels // reduction), out_channels=in_channels, kernel_size=1, stride=1, bias=False),
            h_sigmoid()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        """
        Forward pass of the Squeeze-and-Excitation block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying squeeze-and-excitation.
        """
        b, c, _, _ = x.size()
        y = self.avgpool(x)
        y = self.block(y)
        return x * y

class InvertedResidual(nn.Module):
    """
    Inverted Residual block with depthwise separable convolutions.
    """
    def __init__(self, in_channels, exp_size, out_channels, kernel_size, stride, use_SE, activation):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.h_swish = activation == 'HS'

        if exp_size == in_channels:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False, groups=in_channels),
                nn.BatchNorm2d(num_features=exp_size),
                SE(in_channels=exp_size) if use_SE else nn.Sequential(),
                h_swish() if self.h_swish else nn.ReLU6(),
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=exp_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=exp_size),
                h_swish() if self.h_swish else nn.ReLU6(),
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False, groups=exp_size),
                nn.BatchNorm2d(num_features=exp_size),
                SE(in_channels=exp_size) if use_SE else nn.Sequential(),
                h_swish() if self.h_swish else nn.ReLU6(),
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=out_channels),
            )

    def forward(self, x):
        """
        Forward pass of the Inverted Residual block.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after applying inverted residual block.
        """
        out = self.block(x)
        if self.use_residual:
            out += x
        return out


class MobileNetV3_Large(nn.Module):
    """
    MobileNetV3 Large model architecture.
    """
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Initial convolution block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.hswish = h_swish()

        # Define Inverted Residual blocks
        self.block1 = InvertedResidual(in_channels=16, exp_size=16, out_channels=16, kernel_size=3, stride=1, use_SE=False, activation='RE')
        self.block2 = InvertedResidual(in_channels=16, exp_size=64, out_channels=24, kernel_size=3, stride=2, use_SE=False, activation='RE')
        self.block3 = InvertedResidual(in_channels=24, exp_size=72, out_channels=24, kernel_size=3, stride=1, use_SE=False, activation='RE')
        self.block4 = InvertedResidual(in_channels=24, exp_size=72, out_channels=40, kernel_size=5, stride=2, use_SE=True, activation='RE')
        self.block5 = InvertedResidual(in_channels=40, exp_size=120, out_channels=40, kernel_size=5, stride=1, use_SE=True, activation='RE')
        self.block6 = InvertedResidual(in_channels=40, exp_size=120, out_channels=40, kernel_size=5, stride=1, use_SE=True, activation='RE')
        self.block7 = InvertedResidual(in_channels=40, exp_size=240, out_channels=80, kernel_size=3, stride=2, use_SE=False, activation='HS')
        self.block8 = InvertedResidual(in_channels=80, exp_size=200, out_channels=80, kernel_size=3, stride=1, use_SE=False, activation='HS')
        self.block9 = InvertedResidual(in_channels=80, exp_size=184, out_channels=80, kernel_size=3, stride=1, use_SE=False, activation='HS')
        self.block10 = InvertedResidual(in_channels=80, exp_size=184, out_channels=80, kernel_size=3, stride=1, use_SE=False, activation='HS')
        self.block11 = InvertedResidual(in_channels=80, exp_size=480, out_channels=112, kernel_size=3, stride=1, use_SE=True, activation='HS')
        self.block12 = InvertedResidual(in_channels=112, exp_size=672, out_channels=112, kernel_size=3, stride=1, use_SE=True, activation='HS')
        self.block13 = InvertedResidual(in_channels=112, exp_size=672, out_channels=160, kernel_size=5, stride=2, use_SE=True, activation='HS')
        self.block14 = InvertedResidual(in_channels=160, exp_size=960, out_channels=160, kernel_size=5, stride=1, use_SE=True, activation='HS')
        self.block15 = InvertedResidual(in_channels=160, exp_size=960, out_channels=160, kernel_size=5, stride=1, use_SE=True, activation='HS')

        # Final convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=960, kernel_size=1, stride=1, padding=0, bias=False, groups=160),
            nn.BatchNorm2d(num_features=960),
            h_swish(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            h_swish(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        """
        Forward pass of the MobileNetV3 Large model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after passing through the MobileNetV3 model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.hswish(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)

        x = self.conv2(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
    
        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    model = MobileNetV3_Large().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))
