import torch
import torch.nn as nn

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
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
        self.relu = nn.ReLU()

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
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
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
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
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
    
class InceptionF7(nn.Module):  
    def __init__(self, in_channels, num_conv_1, num_conv_3, num_conv_3_large, num_conv_3_large_out, num_pool_proj):
        super().__init__()

        self.block_1 = ConvBlock(in_channels=in_channels, out_channels=num_conv_1, kernel_size=1, stride=1, padding=0)

        self.block_pool_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=in_channels, out_channels=num_pool_proj, kernel_size=3, stride=1, padding=1)
        )   
        
        self.block_1_3_small_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3, out_channels=num_conv_3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
        )

        self.block_1_3_small_2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3, out_channels=num_conv_3, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        ) 

        self.block_1_3_large_1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3_large, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3_large, out_channels=num_conv_3_large_out, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=num_conv_3_large_out, out_channels=num_conv_3_large_out, kernel_size=(1, 3), stride=1, padding=(0, 1)),

        )

        self.block_1_3_large_2 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3_large, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3_large, out_channels=num_conv_3_large_out, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=num_conv_3_large_out, out_channels=num_conv_3_large_out, kernel_size=(3, 1), stride=1, padding=(1, 0)),

        ) 

    def forward(self, x):
        out_1 = self.block_1(x)
        out_2 = self.block_pool_1(x)

        out_3_1 = self.block_1_3_small_1(x)
        out_3_2 = self.block_1_3_small_2(x)
        out_3 = torch.cat([out_3_1, out_3_2], dim = 1)

        out_4_1 = self.block_1_3_large_1(x)
        out_4_2 = self.block_1_3_large_2(x)
        out_4 = torch.cat([out_4_1, out_4_2], dim = 1)


        x = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return x
    
class InceptionF10(nn.Module):
    def __init__(self, in_channels, num_conv_3, num_conv_1_3_3_out, num_conv_1_3_out):
        super().__init__()

        self.conv_1_3_3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3, out_channels=num_conv_1_3_3_out, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=num_conv_1_3_3_out, out_channels=num_conv_1_3_3_out, kernel_size=3, stride=2, padding=0),
        )

        self.conv_1_3 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=num_conv_3, kernel_size=1, stride=1, padding=0),
            ConvBlock(in_channels=num_conv_3, out_channels=num_conv_1_3_out, kernel_size=3, stride=2, padding=0),
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        )

    def forward(self, x):
        out_1 = self.conv_1_3(x)
        out_2 = self.conv_1_3_3(x)
        out_3 = self.pool(x)

        x = torch.cat([out_1, out_2, out_3], dim=1)

        return x

class InceptionV2(nn.Module):    
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()

        self.block = nn.Sequential(        
            ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=0),
            ConvBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            ConvBlock(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=0),
            ConvBlock(in_channels=80, out_channels=192, kernel_size=3, stride=2, padding=0),
            ConvBlock(in_channels=192, out_channels=288, kernel_size=3, stride=1, padding=1),   # Correction. In paper it is written a normal conv but it must be padded to maintain the input size for the next layer
        )

        self.inception3a = InceptionF5(in_channels=288, num_conv_1=64, num_conv_3_reduce=48, num_conv_3=64, num_conv_5_reduce=64, num_conv_5=96, num_pool_proj=64)

        self.inception3b = InceptionF5(in_channels=288, num_conv_1=64, num_conv_3_reduce=48, num_conv_3=64, num_conv_5_reduce=64, num_conv_5=96, num_pool_proj=64)

        self.inception3c = InceptionF5(in_channels=288, num_conv_1=64, num_conv_3_reduce=48, num_conv_3=64, num_conv_5_reduce=64, num_conv_5=96, num_pool_proj=64)

        self.inceptionpool1 = InceptionF10(in_channels=288, num_conv_3=64, num_conv_1_3_3_out=178, num_conv_1_3_out=302)

        self.inception4a = InceptionF6(in_channels=768, num_conv_1=192, num_conv_7_small=128, num_conv_7_small_out=192, num_conv_7_large=128, num_conv_7_large_out=192, num_pool_proj=192)

        self.inception4b = InceptionF6(in_channels=768, num_conv_1=192, num_conv_7_small=160, num_conv_7_small_out=192, num_conv_7_large=160, num_conv_7_large_out=192, num_pool_proj=192)

        self.inception4c = InceptionF6(in_channels=768, num_conv_1=192, num_conv_7_small=160, num_conv_7_small_out=192, num_conv_7_large=160, num_conv_7_large_out=192, num_pool_proj=192)

        self.inception4d = InceptionF6(in_channels=768, num_conv_1=192, num_conv_7_small=160, num_conv_7_small_out=192, num_conv_7_large=160, num_conv_7_large_out=192, num_pool_proj=192)

        self.inception4e = InceptionF6(in_channels=768, num_conv_1=192, num_conv_7_small=192, num_conv_7_small_out=192, num_conv_7_large=192, num_conv_7_large_out=192, num_pool_proj=192)

        self.inceptionpool2 = InceptionF10(in_channels=768, num_conv_3=192, num_conv_1_3_3_out=194, num_conv_1_3_out=318)

        self.aux = AuxiliaryClassifier(in_channels=768, num_classes=num_classes)

        self.inception5a = InceptionF7(in_channels=1280, num_conv_1=192, num_conv_3=384, num_conv_3_large=448, num_conv_3_large_out=384, num_pool_proj=320)

        self.inception5b = InceptionF7(in_channels=2048, num_conv_1=192, num_conv_3=384, num_conv_3_large=448, num_conv_3_large_out=384, num_pool_proj=320)

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = self.block(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.inception3c(x)

        x = self.inceptionpool1(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)

        aux = self.aux(x)

        x = self.inceptionpool2(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x, aux

    

if __name__ == "__main__":
    from torchsummary import summary

    # Instantiate the InceptionV2 model and move it to the GPU
    model = InceptionV2().to('cuda')

    # Print a summary of the model architecture
    print(summary(model, (3, 299, 299)))

    # print(model)









        
