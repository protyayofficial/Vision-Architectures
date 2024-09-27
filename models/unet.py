import torch
import torch.nn as nn
import torchvision.transforms.functions as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_features)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downsample.append(
                DoubleConv(in_channels=in_channels, out_channels=feature)
            )
            in_channels = feature

        for feature in reversed(features):
            self.upsample.append(
                nn.ConvTranspose2d(in_channels=feature * 2, out_channels=feature, kernel_size=2, stride=2)
            )
            self.upsample.append(
                DoubleConv(in_channels=feature * 2, out_channels=feature)
            )

        self.straight = DoubleConv(in_channels=features[-1], out_channels=features[-1] * 2)

        self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skip_conn = []

        for down in self.downsample:
            x = down(x)
            skip_conn.append(x)
            x = self.maxpool(x)

        x = self.straight(x)
        skip_conn = skip_conn[::-1]

        for idx in range(0, len(self.upsample), step=2):
            x = self.upsample[idx](x)
            skip = skip_conn[idx // 2]

            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])

            concat_skip_con = torch.cat((skip, x), dim=1)
            x = self.upsample[idx + 1](concat_skip_con)

        x = self.final_conv(x)

        return x

if __name__ == "__main__":
    from torchsummary import summary
    model = UNet().to('cuda')
    print(summary(model, (1, 160, 160)))