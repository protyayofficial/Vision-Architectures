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
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, reduction_factor=0.25, survival_prob=0.8):
        super().__init__()

        self.survival_prob = survival_prob

        self.use_residual = (stride == 1 and in_channels == out_channels)

        intermediate_channels = in_channels * expansion_factor

        self.expand = in_channels != intermediate_channels

        intermediate_dims = int(in_channels * reduction_factor)

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
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob

        return torch.div(x, self.survival_prob) * binary_tensor


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
    
class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob
        intermediate_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expansion = ConvBlock(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size // 2))

        self.pointwise = ConvBlock(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def stochastic_depth(self, x):
        if not self.training:
            return x
        
        binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob

        return torch.div(x, self.survival_prob) * binary_tensor

    def forward(self, x):
        residual = x

        x = self.expansion(x)
        x = self.pointwise(x)

        if self.use_residual:
            x = self.stochastic_depth(x)
            x += residual

        return x
    
class EfficientNetv2_s(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, dropout_rate=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        # [expansion factor, output channels, repeats, stride, kernel_size]
        self.stages = [
            [1, 24, 2, 1, 3],
            [4, 48, 4, 2, 3],
            [4, 64, 4, 2, 3],
            [4, 128, 6, 2, 3],
            [6, 160, 9, 1, 3],
            [6, 256, 15, 2, 3],
        ]

        self.features = self._create_layers()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifer = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(in_features=1280, out_features=num_classes)
        )

    def _create_layers(self):

        layers = [
            ConvBlock(
                in_channels=self.in_channels,
                out_channels=24,
                kernel_size=3, 
                stride=2,
                padding=1
            )
        ]

        in_channels = 24

        for stage, (expansion_factor, output_channels, repeats, stride, kernel_size) in enumerate(self.stages):
            for layer in range(repeats):
                if stage+1 in [1, 2, 3]:
                    layers.append(
                        FusedMBConv(
                            in_channels=in_channels,
                            out_channels=output_channels,
                            kernel_size=kernel_size,
                            stride=stride if layer==0 else 1,
                            expansion_factor=expansion_factor
                        )
                    )
                else:
                    layers.append(
                        MBConv(
                            in_channels=in_channels,
                            out_channels=output_channels,
                            kernel_size=kernel_size,
                            stride=stride if layer==0 else 1,
                            expansion_factor=expansion_factor
                        )
                    )
                in_channels = output_channels

        layers.append(
            ConvBlock(
                in_channels=in_channels,
                out_channels=1280,
                kernel_size=1, 
                stride=1,
                padding=0
            )
        )

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)

        return x

if __name__ == "__main__":
    from torchsummary import summary

    model = EfficientNetv2_s().to("cuda")
    summary(model, (3, 300, 300))
    # print(model)