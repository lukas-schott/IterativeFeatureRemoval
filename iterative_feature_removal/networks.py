import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from iterative_feature_removal import utils as u


def get_model(config):
    if config.model == 'CNN' or config.model == 'cnn':
        return VanillaCNN()
    elif config.model == 'MLP' or config.model == 'mlp':
        return MLP()
    else:
        raise Exception(f'Model {config.model} is not defined')


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        print('MLP')
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, X):
        X = X.flatten(1)
        X = F.relu(self.linear1(X))
        X = F.relu(self.linear2(X))
        X = self.linear3(X)
        return X

# class VanillaCNN(nn.Module):
#     def __init__(self):
#         print('vanilla CNN 2')
#         super(VanillaCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 20, 5, 1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


class VanillaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        print('vanilla CNN 1')
        self.conv_1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.relu_1 = nn.ReLU()
        self.pool_1 =nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_2 = nn.Conv2d(16, 16, kernel_size=(4, 4))
        self.relu_2 = nn.ReLU()
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv_3 = nn.Conv2d(16, 120, kernel_size=(4, 4))
        self.relu_3 = nn.ReLU()

        self.fc_4 = nn.Linear(120, 84)
        self.relu_4 = nn.ReLU()
        self.fc_5 = nn.Linear(84, 10)

    def forward(self, img, return_all=False):
        layer_1 = self.pool_1(self.relu_1(self.conv_1(img)))
        layer_2 = self.pool_2(self.relu_2(self.conv_2(layer_1)))
        layer_3 = self.relu_3(self.conv_3(layer_2))
        layer_4 = self.relu_4(self.fc_4(layer_3.flatten(1)))
        logits = self.fc_5(layer_4)
        if return_all:
            return logits, [layer_1, layer_2, layer_3, layer_4]
        return logits
