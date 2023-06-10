import torch
import torch.nn as nn
class AudioDenoisingCNN(nn.Module):
    def __init__(self):
        super(AudioDenoisingCNN, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        self.trans_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.trans_conv2 = nn.ConvTranspose1d(32, 16, kernel_size=5, stride=1, padding=2)
        self.trans_conv3 = nn.ConvTranspose1d(16, 1, kernel_size=7, stride=1, padding=3)

        self.skip_conv1 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.skip_conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.batch_norm3 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x_conv_1 = self.relu(self.batch_norm1(self.conv1(x)))
        x_conv_2 = self.relu(self.batch_norm2(self.conv2(x_conv_1)))
        x_conv_3 = self.relu(self.batch_norm3(self.conv3(x_conv_2)))

        x_skip_1 = self.relu(self.batch_norm2(self.trans_conv1(x_conv_3))) + x_conv_2
        x_skip_2 = self.relu(self.batch_norm1(self.trans_conv2(x_skip_1))) + x_conv_1
        x_tconv_3 = self.trans_conv3(x_skip_2)

        return x_tconv_3