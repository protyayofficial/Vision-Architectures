import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    Implementation of the AlexNet architecture for image classification.

    Args:
        num_classes (int): Number of classes for the final classification layer. Default is 1000 for ImageNet.
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # First convolutional layer followed by ReLU activation and Local Response Normalization (LRN)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.lrn = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)

        # Second convolutional layer with padding to maintain spatial dimensions
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)

        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)

        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)

        # ReLU activation function and max pooling layer
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)

        # Flattened size after the last convolutional layer
        # This is based on the assumption that the input image size is 227x227
        self.flattened_size = 256 * 6 * 6

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=self.flattened_size, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        """
        Defines the forward pass of the AlexNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 227, 227).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Apply first conv layer, ReLU, LRN, and max pooling
        x = self.lrn(self.relu(self.conv1(x)))
        x = self.maxpool(x)

        # Apply second conv layer, ReLU, LRN, and max pooling
        x = self.lrn(self.relu(self.conv2(x)))
        x = self.maxpool(x)

        # Apply third, fourth, and fifth conv layers with ReLU
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        # Apply final max pooling
        x = self.maxpool(x)

        # Flatten the tensor to feed it into the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply first fully connected layer with dropout
        x = self.fc1(x)
        x = self.dropout(x)

        # Apply second fully connected layer with dropout
        x = self.fc2(x)
        x = self.dropout(x)

        # Apply the final fully connected layer to get output logits
        x = self.fc3(x)

        return x
    

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the AlexNet model and move it to the GPU
    model = AlexNet().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 227, 227)))  
