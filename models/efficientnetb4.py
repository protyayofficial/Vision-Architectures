import torch
import torch.nn as nn
import math

class ConvBlock(nn.Module):
    """
    A convolutional block consisting of a Conv2d layer followed by BatchNorm and SiLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups, 
            bias=bias
        )

        self.bn = nn.BatchNorm2d(num_features=out_channels)

        self.act = nn.SiLU(inplace=True)  # Swish activation

    def forward(self, x):
        """
        Forward pass through the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the ConvBlock.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block to perform channel-wise feature recalibration.
    """
    def __init__(self, in_channels, intermediate_dims):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),

            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=intermediate_dims, 
                kernel_size=1
            ),

            nn.SiLU(inplace=True),

            nn.Conv2d(
                in_channels=intermediate_dims, 
                out_channels=in_channels, 
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass through the Squeeze-and-Excitation block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Scaled tensor after applying the squeeze-and-excitation block.
        """

        return x * self.se(x)

class MBConv(nn.Module):
    """
    Mobile Inverted Residual Block (MBConv) with depthwise separable convolution, squeeze-and-excitation, and optional expansion.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, reduction_factor=4, survival_prob=0.8):
        super().__init__()

        self.survial_prob = survival_prob

        self.use_residual = (stride == 1 and in_channels == out_channels)

        intermediate_channels = in_channels * expansion_factor

        self.expand = in_channels != intermediate_channels

        intermediate_dims = int(in_channels // reduction_factor)

        if self.expand:
            self.expansion = ConvBlock(
                in_channels=in_channels, 
                out_channels=intermediate_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
            

        self.block = nn.Sequential(
            ConvBlock(
                in_channels=intermediate_channels, 
                out_channels=intermediate_channels, 
                kernel_size=kernel_size, stride=stride, 
                padding=(kernel_size - 1) // 2, 
                groups=intermediate_channels
            ),
            
            SqueezeExcitation(
                in_channels=intermediate_channels, 
                intermediate_dims=intermediate_dims
            ),

            nn.Conv2d(
                in_channels=intermediate_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=False
            ),

            nn.BatchNorm2d(num_features=out_channels)
        )

    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survial_prob

        return torch.div(x, self.survial_prob) * binary_tensor


    def forward(self, x):
        """
        Forward pass through the MBConv block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the MBConv block.
        """
        residual = x
        if self.expand:
            x = self.expansion(x)
        x = self.block(x)
        if self.use_residual:
            x = self.stochastic_depth(x)
            x += residual
        return x

class EfficientNetB4(nn.Module):
    """
    EfficientNetB4 model with variable depth and width scaling.
    """
    def __init__(self, in_channels=3, phi=3, alpha=1.2, beta=1.1, num_classes=1000, dropout_rate=0.4):
        super().__init__()

        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.in_channels = in_channels

         # [expansion factor, output channels, repeats, stride, kernel_size]
        self.stages = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]

        width_factor, depth_factor = self.calculate_coeff()

        last_channels = math.ceil(1280 * width_factor)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.features = self.create_features(
            width_factor=width_factor, 
            depth_factor=depth_factor, 
            last_channels=last_channels
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=last_channels, out_features=num_classes)
        )

        
    def create_features(self, width_factor, depth_factor, last_channels):
        channels = int(32 * width_factor)

        features = [
            ConvBlock(in_channels=self.in_channels, out_channels=channels, kernel_size=3, stride=2, padding=1),
        ]

        in_channels = channels

        for expansion_factor, channels, repeats, stride, kernel_size in self.stages:
            out_channels = 4 * math.ceil(int(channels * width_factor) / 4)  #in order to keep it divisible by reduction factor

            layer_repeat = math.ceil(repeats * depth_factor)

            for layer in range(layer_repeat):
                features.append(
                    MBConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride if layer==0 else 1,
                        expansion_factor=expansion_factor,
                    )
                )

                in_channels=out_channels

        features.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=last_channels,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        return nn.Sequential(*features)
        

    def calculate_coeff(self):
        width_coefficient = self.beta ** self.phi
        depth_coefficient = self.alpha ** self.phi

        return width_coefficient, depth_coefficient


    def forward(self, x):
        """
        Forward pass through the EfficientNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after passing through the EfficientNet model.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = EfficientNetB4().to("cuda")
    summary(model, (3, 380, 380))
