import torch
import torch.nn as nn

# ResNet

# no auto padding option in pytorch
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(self, Conv2dAuto).__init__(self, *args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


from functools import partial

# define 3x3 convolution with auto padding
conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

# define activation function with ModuleDict for convenience and scalability

def activation_f(activation: str):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()],
    ])[activation]


# define extendable base Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation='relu'):
        super(self, ResidualBlock).__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_f(activation)
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual # f(x) + x
        x = self.activate(x)
        return x


    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels: int, out_channels: int, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super(self, ResNetResidualBlock).__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.expanded_channels)
        ) if self.should_apply_shortcut else None
    

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels: int, out_channels: int, conv, *args, **kwargs):
    return nn.Sequential(
            conv(in_channels, out_channels, *args, **kwargs),
            nn.BatchNorm2d(out_channels)
            )

class ResNetBasicBlock(ResNetResidualBlock):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, vias=False),
            activation_f(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, bias=False)
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation_f(self.activation),
            conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
            activation_f(self.activation),
            conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1)
        )


class ResNetLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
           block(in_channels, out_channels, *args, **kwargs, downsampling=downsampling),
           *[block(out_channels * block.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )
    def forward(self, x):
        return self.blocks(x)


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2,2,2,2], activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_f(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
       
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([
            ResNetLayer(block_sizes[0], block_sizes[0], n=depths[0], activation=activation, block=block, *args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, out_channels, n=n, activation=activation, block=block, *args, **kwargs) for (in_channels, out_channels), n in zip(self.in_out_blocks_sizes, depths[1:])]
        ])

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x

class ResNetDecoder(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResNedDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

def resnet34(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[3, 4, 6, 3], *args, **kwargs)
