import torch.nn as nn
from collections import OrderedDict
from iterative_feature_removal import utils as u


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 16, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            # ('b1', nn.BatchNorm2d(16, affine=False)),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(16, 16, kernel_size=(5, 5))),
            # ('b2', nn.BatchNorm2d(16, affine=False)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(4, 4))),
            ('relu5', nn.ReLU()),
            # ('b3', nn.BatchNorm2d(120, affine=False)),
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            # ('b3', nn.BatchNorm1d(84, affine=False)),
            ('f7', nn.Linear(84, 10)),
        ]))

    def forward(self, img):
        u.check_bounds(img, 0, 1)
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
