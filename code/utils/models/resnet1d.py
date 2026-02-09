import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class MyConv1dPadSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=False
        )

    def forward(self, x):
        net = x
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)
        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net

class BasicBlock_1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, downsample, use_do):
        super(BasicBlock_1D, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.stride)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1)

        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu2(out)


class ResNet1D(nn.Module):
    def __init__(self, in_channels, out_dim, layer_num=[2, 2, 2, 2]):
        super(ResNet1D, self).__init__()
        self.init_conv = nn.Conv1d(in_channels, 32, kernel_size=7, stride=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.channels = 32

        layers = []
        for _ in range(layer_num[0] ):
            layers.append(BasicBlock_1D(self.channels, self.channels, 3, 1, False, False))
        self.layer1 = nn.Sequential(*layers)
        self.conv12 = nn.Conv1d(self.channels, self.channels * 2, 3, 2, 1, bias=False)

        layers = []
        for _ in range(layer_num[1]):
            layers.append(BasicBlock_1D(self.channels * 2, self.channels * 2, 3, 1, False,  False))
        self.layer2 = nn.Sequential(*layers)
        self.conv23 = nn.Conv1d(self.channels * 2, self.channels * 4, 3, 2, 1, bias=False)

        layers = []
        for _ in range(layer_num[2]):
            layers.append(BasicBlock_1D(self.channels * 4, self.channels * 4, 3, 1, False,  False))
        self.layer3 = nn.Sequential(*layers)

        self.conv34 = nn.Conv1d(self.channels * 4, self.channels * 8, 3, 2, 1, bias=False)
        layers = []
        for _ in range(layer_num[3]):
            layers.append(BasicBlock_1D(self.channels * 8, self.channels * 8, 3, 1, False,  False))
        self.layer4 = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(2)
        self.linear = nn.Linear(self.channels * 8 * 2, out_dim, bias=False)

    def forward(self, x, return_emb=False):
        x = x.squeeze(1)
        x = torch.transpose(x, 1, 2)
        out1 = F.relu(self.bn1(self.init_conv(x)))
        out1 = self.layer1(out1)
        out1 = F.relu(self.conv12(out1))
        out1 = self.layer2(out1)
        out1 = F.relu(self.conv23(out1))
        out1 = self.layer3(out1)
        out1 = F.relu(self.conv34(out1))
        out1 = self.layer4(out1)
        out1 = self.pool(out1)
        out1 = out1.view(out1.size(0), -1)
        out = self.linear(out1)

        if return_emb:
            return out1, out
        else:
            return out


