'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from collections import OrderedDict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10, bias=False)

    def forward(self, x, return_emb=False):
        out = self.features(x)
        #intermediate_out = {}
        #intermediate_out['input'] = x
        #for name, module in self.features.named_children():
        #    x = module(x)
        #    intermediate_out[name] = x

        out = out.view(out.size(0), -1)
        if return_emb:
            return out, self.classifier(out)
        else:
            return self.classifier(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [(f'layer{i}_maxpool', nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                layers += [(f'layer{i}_conv', nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False)),
                           (f'layer{i}_bn', nn.BatchNorm2d(x)),
                           (f'layer{i}_relu', nn.ReLU(inplace=True))]
                in_channels = x
        layers += [(f'layer{i}_avgpool', nn.AvgPool2d(kernel_size=1, stride=1))]
        return nn.Sequential(OrderedDict(layers))


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
