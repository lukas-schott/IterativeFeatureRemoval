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


# model from https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/models/mnist/small_cnn.py
class VanillaCNN(nn.Module):
    def __init__(self, drop=0.5):
        super(VanillaCNN, self).__init__()

        self.num_channels = 1
        self.num_labels = 10

        activ = nn.ReLU(True)

        # feature extractor
        self.conv_1 = nn.Conv2d(self.num_channels, 32, 3)
        self.relu_1 = activ

        self.conv_2 = nn.Conv2d(32, 32, 3)
        self.relu_2 = activ

        self.pool_3 = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(32, 64, 3)
        self.relu_3 = activ

        self.conv_4 = nn.Conv2d(64, 64, 3)
        self.relu_4 = activ

        # classifier
        self.pool_5 = nn.MaxPool2d(2, 2)
        self.fc_5 = nn.Linear(64 * 4 * 4, 200)
        self.relu_5 = activ

        self.drop_5 = nn.Dropout(drop)
        self.fc_6 = nn.Linear(200, 200)
        self.relu_5 = activ

        self.last = nn.Linear(200, self.num_labels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.last.weight, 0)
        nn.init.constant_(self.last.bias, 0)

    def forward(self, input, return_activations=False):
        layer_1 = self.relu_1(self.conv_1(input))
        layer_2 = self.relu_2(self.conv_2(layer_1))
        layer_3 = self.relu_3(self.conv_3(self.pool_3(layer_2)))
        layer_4 = self.relu_4(self.conv_4(layer_3))

        layer_5 = self.relu_5(self.fc_5(self.pool_5(layer_4).view(-1, 64 * 4 * 4)))
        logits = self.last(layer_5)
        if return_activations:
            return logits, (layer_1, layer_2, layer_3, layer_4, layer_5)
        return logits
