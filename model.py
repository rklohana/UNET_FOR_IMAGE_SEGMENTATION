"""Assignment-style U-Net with valid 3x3 convolutions and cropped skip connections."""

import torch
from torch import nn


def center_crop(tensor, height, width):
    """Crop the spatial center of an image, mask, or feature map."""
    source_height, source_width = tensor.shape[-2:]
    if height > source_height or width > source_width:
        raise ValueError(f"Cannot crop {source_height}x{source_width} to {height}x{width}")
    top = (source_height - height) // 2
    left = (source_width - width) // 2
    return tensor[..., top:top + height, left:left + width]


class twoConvBlock(nn.Module):
    """Valid convolution, ReLU, valid convolution, batch norm, ReLU."""

    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channel, output_channel, kernel_size=3),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class downStep(nn.Module):
    """Contracting path: four conv/pool stages and one bottleneck block."""

    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.blocks = nn.ModuleList(
            [twoConvBlock(1, c), twoConvBlock(c, 2*c),
             twoConvBlock(2*c, 4*c), twoConvBlock(4*c, 8*c)]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = twoConvBlock(8*c, 16*c)

    def forward(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return self.bottleneck(x), skips


class upStep(nn.Module):
    """Transpose convolutions followed by cropped skip concatenation."""

    def __init__(self, base_channels=64):
        super().__init__()
        c = base_channels
        self.upsamples = nn.ModuleList(
            [nn.ConvTranspose2d(16*c, 8*c, 2, stride=2),
             nn.ConvTranspose2d(8*c, 4*c, 2, stride=2),
             nn.ConvTranspose2d(4*c, 2*c, 2, stride=2),
             nn.ConvTranspose2d(2*c, c, 2, stride=2)]
        )
        self.blocks = nn.ModuleList(
            [twoConvBlock(16*c, 8*c), twoConvBlock(8*c, 4*c),
             twoConvBlock(4*c, 2*c), twoConvBlock(2*c, c)]
        )

    def forward(self, x, skips):
        for upsample, block, skip in zip(self.upsamples, self.blocks, reversed(skips)):
            x = upsample(x)
            skip = center_crop(skip, *x.shape[-2:])
            x = block(torch.cat((skip, x), dim=1))
        return x


class UNet(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        self.down = downStep(base_channels)
        self.up = upStep(base_channels)
        self.classifier = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x):
        x, skips = self.down(x)
        return self.classifier(self.up(x, skips))  # Raw logits for CrossEntropyLoss.
