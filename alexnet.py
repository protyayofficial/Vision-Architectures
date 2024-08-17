import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes = 1000):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding = 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding = 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding = 1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding = 1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(0.5)

        self.flattened_size = 256 * 6 * 6

        self.fc1 = nn.Linear(in_features=self.flattened_size, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        x = self.lrn(self.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.lrn(self.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
