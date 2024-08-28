import torch
import torch.nn as nn

class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary Classifier used within the GoogLeNet architecture.
    Provides intermediate supervision during training, helping to combat vanishing gradients.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.

    Methods:
        forward(x):
            Performs the forward pass through the auxiliary classifier.
    """
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        """
        Forward pass through the AuxiliaryClassifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after passing through the auxiliary classifier.
        """
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(in_place=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class InceptionF5(nn.Module):  
    def __init__(self, in_channels, num_conv_1, num_conv_3_reduce, num_conv_3, num_conv_5_reduce, num_conv_5, num_pool_proj):
        super().__init__()

        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=num_conv_1, kernel_size=1, stride=1, padding=0)
        
        self.block_1_3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3_reduce, out_channels=num_conv_3, kernel_size=3, stride=1, padding=1)
        )

        self.block_pool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=num_pool_proj, kernel_size=3, stride=1, padding=1)
        )       

        self.block_1_3_3 =  nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_5_reduce, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_5_reduce, out_channels=num_conv_5, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=num_conv_5, out_channels=num_conv_5, kernel_size=3, stride=1, padding=1),
            
            
        )

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_pool_1(x)
        out_3 = self.block_1_3(x)
        out_4 = self.block_1_3_3(x)


        x = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return x
    
class InceptionF6(nn.Module):  
    def __init__(self, in_channels, num_conv_1, num_conv_7_small, num_conv_7_small_out, num_conv_7_large, num_conv_7_large_out, num_pool_proj):
        super().__init__()

        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=num_conv_1, kernel_size=1, stride=1, padding=0)

        self.block_pool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=num_pool_proj, kernel_size=3, stride=1, padding=1)
        )   
        
        self.block_1_3_small_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_7_small, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_7_small, out_channels=num_conv_7_small, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_small, out_channels=num_conv_7_small_out, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

            

        self.block_1_3_large =  nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_7_large, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large_out, kernel_size=(7, 1), stride=1, padding=(3, 0)),            
        )

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_pool_1(x)
        out_3 = self.block_1_7_small(x)
        out_4 = self.block_1_7_large(x)


        x = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return x
    
class InceptionF7(nn.Module):  
    def __init__(self, in_channels, num_conv_1, num_conv_7_small, num_conv_7_small_out, num_conv_7_large, num_conv_7_large_out, num_pool_proj):
        super().__init__()

        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=num_conv_1, kernel_size=1, stride=1, padding=0)

        self.block_pool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=num_pool_proj, kernel_size=3, stride=1, padding=1)
        )   
        
        self.block_1_7_small = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_7_small, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_7_small, out_channels=num_conv_7_small, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_small, out_channels=num_conv_7_small_out, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

            

        self.block_1_7_large =  nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_7_large, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            ConvBlock(in_channels=num_conv_7_large, out_channels=num_conv_7_large_out, kernel_size=(7, 1), stride=1, padding=(3, 0)),            
        )

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_pool_1(x)
        out_3 = self.block_1_7_small(x)
        out_4 = self.block_1_7_large(x)


        x = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return x

class InceptionV2(nn.Module):
    """
    InceptionV2 architecture implementation in PyTorch.

    This implementation includes auxiliary classifiers as per the original 
    paper to help with convergence during training. The auxiliary classifiers 
    can be disabled during inference.

    Attributes:
    -----------
    aux_classifiers : bool
        Whether to use auxiliary classifiers during training.
    convblock1 : nn.Sequential
        The first convolutional block consisting of a convolutional layer 
        followed by ReLU activation.
    convblock2 : nn.Sequential
        The second convolutional block consisting of a convolutional layer 
        followed by ReLU activation.
    maxpool : nn.MaxPool2d
        Max pooling layer.
    avgpool : nn.AvgPool2d
        Average pooling layer for the final global average pooling.
    dropout : nn.Dropout
        Dropout layer applied before the final fully connected layer.
    linear : nn.Linear
        Final fully connected layer for classification.
    aux_4a : AuxiliaryClassifier
        First auxiliary classifier connected after the first inception block in the 4th layer.
    aux_4d : AuxiliaryClassifier
        Second auxiliary classifier connected after the third inception block in the 4th layer.
    inception3a : InceptionBlock
        First inception block in the 3rd layer.
    inception3b : InceptionBlock
        Second inception block in the 3rd layer.
    inception4a : InceptionBlock
        First inception block in the 4th layer.
    inception4b : InceptionBlock
        Second inception block in the 4th layer.
    inception4c : InceptionBlock
        Third inception block in the 4th layer.
    inception4d : InceptionBlock
        Fourth inception block in the 4th layer.
    inception4e : InceptionBlock
        Fifth inception block in the 4th layer.
    inception5a : InceptionBlock
        First inception block in the 5th layer.
    inception5b : InceptionBlock
        Second inception block in the 5th layer.

    Methods:
    --------
    forward(x):
        Defines the forward pass of the GoogLeNet model.
    """
    
    def __init__(self, aux_classifiers=True, num_classes=1000):
        """
        Initializes the InceptionV2 model with the option to use auxiliary classifiers.

        Parameters:
        -----------
        aux_classifiers : bool, optional
            If True, includes auxiliary classifiers in the model (default is True).
        num_classes : int, optional
            Number of output classes for the final classification layer (default is 1000).
        """
        super().__init__()

        self.aux_classifiers = aux_classifiers

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU()
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(in_features=1024, out_features=num_classes)

        if self.aux_classifiers:
            self.aux_4a = AuxiliaryClassifier(in_channels=512, num_classes=num_classes)
            self.aux_4d = AuxiliaryClassifier(in_channels=528, num_classes=num_classes)

        self.inception3a = InceptionBlock(
            in_channels=192, num_conv_1=64, num_conv_3_reduce=96, 
            num_conv_3=128, num_conv_5_reduce=16, num_conv_5=32, 
            num_pool_proj=32
        )
        
        self.inception3b = InceptionBlock(
            in_channels=256, num_conv_1=128, num_conv_3_reduce=128, 
            num_conv_3=192, num_conv_5_reduce=32, num_conv_5=96, 
            num_pool_proj=64
        )

        self.inception4a = InceptionBlock(
            in_channels=480, num_conv_1=192, num_conv_3_reduce=96, 
            num_conv_3=208, num_conv_5_reduce=16, num_conv_5=48, 
            num_pool_proj=64
        )

        self.inception4b = InceptionBlock(
            in_channels=512, num_conv_1=160, num_conv_3_reduce=112, 
            num_conv_3=224, num_conv_5_reduce=24, num_conv_5=64, 
            num_pool_proj=64
        )

        self.inception4c = InceptionBlock(
            in_channels=512, num_conv_1=128, num_conv_3_reduce=128, 
            num_conv_3=256, num_conv_5_reduce=24, num_conv_5=64, 
            num_pool_proj=64
        )

        self.inception4d = InceptionBlock(
            in_channels=512, num_conv_1=112, num_conv_3_reduce=144, 
            num_conv_3=288, num_conv_5_reduce=32, num_conv_5=64, 
            num_pool_proj=64
        )

        self.inception4e = InceptionBlock(
            in_channels=528, num_conv_1=256, num_conv_3_reduce=160, 
            num_conv_3=320, num_conv_5_reduce=32, num_conv_5=128, 
            num_pool_proj=128
        )

        self.inception5a = InceptionBlock(
            in_channels=832, num_conv_1=256, num_conv_3_reduce=160, 
            num_conv_3=320, num_conv_5_reduce=32, num_conv_5=128, 
            num_pool_proj=128
        )

        self.inception5b = InceptionBlock(
            in_channels=832, num_conv_1=384, num_conv_3_reduce=192, 
            num_conv_3=384, num_conv_5_reduce=48, num_conv_5=128, 
            num_pool_proj=128
        )

    def forward(self, x):
        """
        Defines the forward pass of the GoogLeNet model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, 224, 224).

        Returns:
        --------
        torch.Tensor
            Output logits for the main classifier.
        torch.Tensor, torch.Tensor, optional
            Output logits for the auxiliary classifiers if used.
        """
        
        aux1, aux2 = None, None

        x = self.convblock1(x)
        x = self.maxpool(x)

        x = self.convblock2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)

        if self.aux_classifiers and self.training:
            aux1 = self.aux_4a(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_classifiers and self.training:
            aux2 = self.aux_4d(x)

        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)

        if self.aux_classifiers and self.training:
            return x, aux1, aux2
        else:
            return x
    

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the InceptionV2 model and move it to the GPU
    model = InceptionV2().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 224, 224)))









        
