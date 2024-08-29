import torch
import torch.nn as nn

class VGG16(nn.Module):
    """
    Implementation of the VGG16 architecture for image classification.

    Args:
        num_classes (int): Number of classes for the final classification layer. Default is 1000 for ImageNet.
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # First block of convolutional layers followed by Max Pooling
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second block of convolutional layers followed by Max Pooling
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third block of convolutional layers followed by Max Pooling
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fourth block of convolutional layers followed by Max Pooling
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fifth block of convolutional layers followed by Max Pooling
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # ReLU activation and Dropout for the fully connected layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        # Initialize the in_features for the fully connected layers by passing a dummy input through the network
        self._initialize_fc()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def _initialize_fc(self):
        """
        Initializes the in_features size for the first fully connected layer.
        This is done by passing a dummy input through the convolutional layers
        to determine the output size.
        """
        # Create a dummy input with the shape of (1, 3, 224, 224)
        dummy_input = torch.zeros(1, 3, 224, 224)

        # Pass the dummy input through all convolutional blocks
        out = self.block1(dummy_input)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)

        # Flatten the output to calculate the number of input features for the first fully connected layer
        self.in_features = out.view(-1).size(0)

    def forward(self, x):
        """
        Defines the forward pass of the VGG16 model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Pass through all convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Flatten the output to feed it into the fully connected layers
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers with ReLU and Dropout
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer with Softmax activation
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the VGG16 model and move it to the GPU
    model = VGG16().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))
