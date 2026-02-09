'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1   = nn.Linear(16*5*5, 120, bias=False)
        self.fc2   = nn.Linear(120, 84, bias=False)
        self.classifier   = nn.Linear(84, num_classes, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.classifier(out)
        return out


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes, in_channel=1):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=5, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        if in_channel == 3:
            self.fc = nn.Linear(400, 120, bias=False)
        elif in_channel == 1:
            self.fc = nn.Linear(256, 120, bias=False)
        else:
            raise NotImplementedError(f'in_channel={in_channel} not implemented')
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84, bias=False)
        self.relu1 = nn.ReLU()
        self.classifier = nn.Linear(84, num_classes, bias=False)

    def forward(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.relu(self.fc(out))
        out = self.relu1(self.fc1(out))
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

class LeNet1(nn.Module):
    def __init__(self, num_classes):
        super(LeNet1, self).__init__()
        # input is Nx1x28x28
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 4, 5, bias=False),
            #nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 12, 5, bias=False),
            #nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.classifier = nn.Linear(12 * 4 * 4, num_classes)

    def forward(self, x, return_emb=False):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(x.size(0), -1)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)


def test():
    from torchsummary import summary
    #net = LeNet1(num_classes=10)
    #summary(net, (1, 28, 28))
    net = LeNet5(num_classes=10).cuda()
    summary(net, (1, 28, 28))
    #net = LeNet5_dwscaled(num_classes=10).cuda()
    #summary(net, (1, 28, 28))

#test()
