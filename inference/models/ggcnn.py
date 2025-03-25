import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel


class GenerativeResnet(GraspModel):

    """
    GG-CNN, RSS Paper (https://arxiv.org/abs/1804.05172)
    """

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=3, padding=3)
        self.conv2 = nn.Conv2d(channel_size, channel_size//2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(channel_size//2, channel_size//4, kernel_size=3, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(channel_size//4, channel_size//4, kernel_size=3, stride=2, padding=1,
                                         output_padding=1)
        self.convt2 = nn.ConvTranspose2d(channel_size//4, channel_size//2, kernel_size=5, stride=2, padding=2,
                                         output_padding=1)
        self.convt3 = nn.ConvTranspose2d(channel_size//2, channel_size, kernel_size=9, stride=3, padding=3,
                                         output_padding=1)

        self.pos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(channel_size, output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.relu(self.convt2(x))
        x = F.relu(self.convt3(x))

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output
