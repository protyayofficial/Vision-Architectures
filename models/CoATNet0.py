import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False, use_act=True):
        super().__init__()

        self.use_act = use_act

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        
        if self.use_act:
            x = self.act(x)

        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction_factor=4):
        super().__init__()

        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_factor, kernel_size=1, stride=1, padding=0, bias=False)
        self.excitation = nn.Conv2d(in_channels=in_channels // reduction_factor, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self, x):
        out = self.avgpool(x)
        out = self.squeeze(out)
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)

        return x * out

class PreActivation(nn.Module):
    def __init__(self, features, function, activation):
        super().__init__()

        self.act = activation(features)
        self.function = function

    def forward(self, x, **kwargs):
        x = self.act(x)
        return self.function(x, **kwargs)

class FFN(nn.Module):
    def __init__(self, in_features, hidden_dims):
        super().__init__()

        self.linear1 = nn.Linear(in_features=in_features, out_features=hidden_dims)
        self.linear2 = nn.Linear(in_features=hidden_dims, out_features=in_features)

        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)

        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=4, use_downsample=False):
        super().__init__()

        self.use_downsample = use_downsample
        stride = 2 if self.use_downsample else 1
        hidden_dims = int(in_channels // expansion_ratio)

        if self.use_downsample:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.layers = []

        if expansion_ratio == 4:
            self.layers.append(
                ConvBlock(in_channels=in_channels, out_channels=hidden_dims, kernel_size=kernel_size, stride=stride, padding=0)
            )

        if expansion_ratio == 1:
            self.layers.append(
                ConvBlock(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=3, stride=stride, padding=1, groups=hidden_dims)
            )
        elif expansion_ratio == 4:
            self.layers.append(
                ConvBlock(in_channels=hidden_dims, out_channels=hidden_dims, kernel_size=3, stride=1, padding=1, groups=hidden_dims)
            )
            self.layers.append(
                SqueezeExcitation(in_channels=hidden_dims)
            )

        self.layers.append(
            ConvBlock(in_channels=hidden_dims, out_channels=out_channels, kernel_size=1, stride=1, padding=0, use_act=False)
        )

        self.conv = PreActivation(features=in_channels, function=nn.Sequential(*self.layers), activation=nn.BatchNorm2d)

    def forward(self, x):
        if self.use_downsample:
            return self.projection(self.maxpool(x)) + self.conv(x)
        else:
            return x + self.conv(x)

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, heads=8, head_dims=32):
        super().__init__()

        hidden_dims = heads * head_dims
        project_out = !(heads == 1 and head_dims == in_channels)

        self.h, self.w = img_size
        self.heads = heads
        self.scale = 1 // (head_dims ** 0.5)

        self.relative_bias = nn.Parameter(
            torch.zeros((2 * self.h - 1, 2 * self.w - 1, heads))
        )

        coords = torch.meshgrid((torch.arange(self.h), torch.arange(self.w)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.h - 1
        relative_coords[1] += self.w - 1
        relative_coords[0] *= 2 * self.w - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.qkv = nn.Linear(in_features=in_channels, out_features=hidden_dims * 3, bias=False)

        self.projection = nn.Sequential(
            nn.Linear(in_features=hidden_dims, out_features=out_channels),
            nn.Dropout(p=0.1)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.qkv(x)
        qkv = qkv.chunk(3, dim=-1)
        q = rearrange('b n (h d) -> b n h d', h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        k_t = k.transpose(-2, -1)

        dot_prod = (q @ k_t) * self.scale

        relative_bias = self.relative_bias.gather(0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h = self.h * self.w, w = self.h * self.w)       dot_prod = dot_prod + relative_bias

        dot_prod = torch.softmax(dot_prod, dim=-1)

        weighted_values = (attn @ v)
        
        weighted_values = rearrange(weighted_values, 'b h n d -> b n (h d)')
        out = self.projection(weighted_values)

        return out


class Transformer(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, heads=8, head_dims=32, use_downsample=False):
        super().__init__()

        self.h, self.w = img_size
        self.use_downsample = use_downsample

        if self.use_downsample:
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.attention = Attention(in_channels=in_channels, out_channels=out_channels, img_size=img_size, heads=heads, head_dims=head_dims)
        self.ffn = FFN(in_features=out_channels, hidden_dims=out_channels * 4)

        self.attention = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            PreActivation(in_channels=in_channels, function=self.attention, activation=nn.LayerNorm)
            Rearrange('b (h w) c -> b c h w', h=self.h, w=self.w)
        )

        self.ffn = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            PreActivation(in_channels=out_channels, function=self.ffn, activation=nn.LayerNorm)
            Rearrange('b (h w) c -> b c h w')
        )

    def forward(self, x):
        if self.use_downsample:
            x = self.projection(self.maxpool1(x)) + self.attention(self.maxpool2(x))
        else:
            x = x + self.attention(x)
        x = x + self.ffn(x)

        return x
