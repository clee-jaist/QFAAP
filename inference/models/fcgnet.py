import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
from inference.models.grasp_model import GraspModel, ResidualBlock

class GenerativeResnet(GraspModel):

    """
    FCG-Net, ICICML Paper (10.1109/ICICML57342.2022.10009877)
    """

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        # Activation function
        self.activation = Mish()
        self.up1 = Upsample(128, 64)  # Upsample layer 1
        self.up2 = Upsample(64, 32)   # Upsample layer 2
        self.up3 = Upsample(32, 32)   # Upsample layer 3
        self.resbody1 = Resblock_body(128, 64)  # Residual block body
        # Convolution layers
        self.conv0 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1)
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res0 = Resblock_body(channel_size * 4, channel_size * 4)
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        # Upsampling layers
        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4, output_padding=0)

        # Output layers for different parameters
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        # Dropout layers
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # Xavier weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = self.activation(x)
        x = self.bn2(self.conv2(x))
        x = self.activation(x)
        x = self.bn3(self.conv3(x))
        x = self.activation(x)

        # Residual block body processing
        x = self.resbody1(x)
        x = self.resbody1(x)
        x = self.resbody1(x)
        x = self.resbody1(x)
        x = self.resbody1(x)
        x = self.res1(x)
        route0 = x
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Adding residual connection
        x = route0 + x

        up1 = self.up1(x)
        x = self.bn4(self.conv4(x))
        x = self.activation(x)
        x = up1 + x

        up2 = self.up2(x)
        up2 = self.conv0(up2)

        # Upsampling and processing
        x = self.bn5(self.conv5(x))
        x = self.activation(x)

        x = self.conv6(x)
        x = torch.add(x, up2)

        # Dropout and output layers
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

# Mish activation function
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# Basic convolution block with BatchNorm and Mish activation
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Residual block with basic convolution layers
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

# Residual block body for feature aggregation
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels
        self.conv1 = BasicConv(in_channels, out_channels, 3)
        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        x = self.conv1(x)
        route = x
        c = self.out_channels
        x = torch.split(x, c // 2, dim=1)[1]
        x = self.conv2(x)
        route1 = x
        x = self.conv3(x)
        x = torch.cat([x, route1], dim=1)
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)

        return x

# Upsample class for upscaling features
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x

# Convolution layer with BatchNorm and LeakyReLU
def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))
