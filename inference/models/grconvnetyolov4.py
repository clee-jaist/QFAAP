import torch
import torch.nn.functional as F

import torch.nn as nn
import math
from collections import OrderedDict
from inference.models.grasp_model import GraspModel, ResidualBlock


class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        # 修改部分-------------------
        # 激活函数------------------------------------
        self.activation = Mish()
        self.up1 = Upsample(128, 64)
        self.up2 = Upsample(64, 32)
        self.up3 = Upsample(32, 32)
        self.resbody1 = Resblock_body(128,64)
        # -----------------------------------
        # bianhuan chidu
        self.conv0 = nn.Conv2d(32, 32, kernel_size=2,stride=1,padding=1)
        # conv1 4--->32 -- kernel_size = 9
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        # 32
        self.bn1 = nn.BatchNorm2d(channel_size)
        # conv2 32--->32*2  kernel_size =4
        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)
        # conv3 64--->128
        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res0 = Resblock_body(channel_size * 4, channel_size * 4)
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)
        # shang caiyang
        # self.up =nn.
        # 128----------> 64
        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=3, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        # 64-------->32  225*225?
        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)
        # 32------->32
        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4,output_padding=0)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        # drop yexu keyi youhua
        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        # x = F.relu(self.bn1(self.conv1(x_in)))
        x = self.bn1(self.conv1(x_in))
        # print(x.shape)
        x = self.activation(x)
        x = self.bn2(self.conv2(x))
        x = self.activation(x)
        x = self.bn3(self.conv3(x))
        x = self.activation(x)



        # gaicheng resblock-body
        # x = F.relu(self.bn1(self.conv1(x_in)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        # x = self.res0(x)
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
        # da canchaibian
        x = route0+x
        # 4,128,56,56
        # print(x.shape)
        # print(x.shape)
        up1 = self.up1(x)
        # print(up1.shape)
        # x==2,64,113,113
        x = self.bn4(self.conv4(x))
        # 4,64,112,112

        # print(x.shape)
        x = self.activation(x)
        x = up1 + x

        # print(x.shape)
        up2 = self.up2(x)
        # print(up2.shape)
        up2 = self.conv0(up2)
        # print('up2', up2.shape)
        # up3 = self.up3(up2)
        # print('up3', up3.shape)
        # print(up2.shape)
        # up2 ---> 2,32,224,224
        # print(up2.shape)
        # x 2,32,225,225
        x = self.bn5(self.conv5(x))
        # print(x.shape)
        x = self.activation(x)
        # x = x + up2
        # print(x.shape)
        # x = F.relu(self.bn4(self.conv4(x)))
        # x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        # print('x', x.shape)
        x = torch.add(x, up2)
        # print(x.shape)

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




# """
# MISH
# jihuohanshu
# """
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


#-------------------------------------------------#
#   卷积块
#   CONV+BATCHNORM+MISH 不需要降维，降尺寸
#-------------------------------------------------#

#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

#---------------------------------------------------#
#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
#---------------------------------------------------#


class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()
        # 传入none表示不需要降维
        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels
        # C B M
        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x
        # 进行通道的二等分 方便后续的contact
        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)
        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)
        feat = x
        x = torch.cat([route, x], dim=1)

        # 利用最大池化进行高和宽的压缩
        # x = self.maxpool(x)
        return x


class Resblock_body1(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            # 上采样 ，拉伸为原来的 两倍。scale_factor
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

# cbl 模块 ==  conv + bn + leakyrelu
# ding yi yi ge conv2d  moudel cbl

def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    # orderdict 字典的模式
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        # 0.1 是a的系数 az
        ("relu", nn.LeakyReLU(0.1)),
    ]))

# xiacaiyang
