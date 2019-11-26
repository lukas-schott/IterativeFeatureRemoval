import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import torchvision.models as models
import math


def get_model(config):
    if config.dataset == 'MNIST':
        if config.model == 'CNN' or config.model == 'cnn':
            # model = VanillaCNN(n_groups=config.n_redundant)
            model = RedundancyNetworks(config.n_redundant)
        elif config.model == 'MLP' or config.model == 'mlp':
            model = MLP()
        else:
            raise Exception(f'Model {config.model} is not defined')
    elif config.dataset == 'greyscale_CIFAR10':
        model = models.resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = torch.nn.Linear(512, 10, bias=True)
    elif config.dataset == 'CIFAR10':
        image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
        model = NormalizedModel(WideResNet(depth=28, num_classes=10,  widen_factor=10, dropRate=0.3),
                                mean=image_mean, std=image_std)
    else:
        raise Exception('no model not available')

    if config.model_load_path != '':
        # print('torch.load(config.model_load_path)', torch.load(config.model_load_path))
        # return torch.load(config.model_load_path)
        model.load_state_dict(torch.load(config.model_load_path)['model'])
        print('model loaded')
    return model


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
    def __init__(self, drop=0.5, n_groups=1):
        super().__init__()

        self.num_channels = 1
        self.num_labels = 10
        self.cached_batches = False

        activ = nn.LeakyReLU(True)
        activ = nn.ReLU(True)
        # activ = nn.LeakyReLU(True)

        # feature extractor
        self.conv_1 = nn.Conv2d(self.num_channels*n_groups, 32*n_groups, 3, groups=n_groups)
        self.relu_1 = activ

        self.conv_2 = nn.Conv2d(32*n_groups, 32*n_groups, 3, groups=n_groups)
        self.relu_2 = activ

        self.pool_3 = nn.MaxPool2d(2, 2)
        self.conv_3 = nn.Conv2d(32*n_groups, 64*n_groups, 3, groups=n_groups)
        self.relu_3 = activ

        self.conv_4 = nn.Conv2d(64*n_groups, 64*n_groups, 3, groups=n_groups)
        self.relu_4 = activ

        # classifier
        self.pool_5 = nn.MaxPool2d(2, 2)
        self.fc_5 = nn.Conv2d(64*n_groups, 200*n_groups, 4, groups=n_groups)
        self.relu_5 = activ

        self.drop_5 = nn.Dropout(drop)
        self.fc_6 = nn.Conv2d(200*n_groups, 200*n_groups, 1, groups=n_groups)
        self.relu_5 = activ

        self.last = nn.Conv2d(200*n_groups, self.num_labels*n_groups, 1, groups=n_groups)
        self.n_groups = n_groups
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # nn.init.constant_(self.last.weight, 0)
        # nn.init.constant_(self.last.bias, 0)

    def forward(self, input, return_activations=False, return_individuals=False, cache_input=True):
        if self.n_groups > 1:
            new_input = input[:, None]
            self.cached_batches = new_input.expand(input.shape[0], self.n_groups, input.shape[1], *input.shape[2:])
            self.cached_batches.requires_grad_(True)
            input = self.cached_batches.view(input.shape[0], self.n_groups * input.shape[1], *input.shape[2:])

        layer_1 = self.relu_1(self.conv_1(input))
        layer_2 = self.relu_2(self.conv_2(layer_1))
        layer_3 = self.relu_3(self.conv_3(self.pool_3(layer_2)))
        layer_4 = self.relu_4(self.conv_4(layer_3))
        layer_5 = self.relu_5(self.fc_5(self.pool_5(layer_4)))
        last_layer = self.last(layer_5)
        assert last_layer.shape[-1] == last_layer.shape[-2] == 1
        individual_logits = last_layer[:, :, 0, 0]   # (b, n_classes)

        if self.n_groups > 1:
            individual_logits = individual_logits.view(input.shape[0], self.n_groups, self.num_labels)
            individual_logits = individual_logits - torch.min(individual_logits, dim=2, keepdim=True).values
            individual_logits /= torch.max(individual_logits, dim=2, keepdim=True).values
            logits = individual_logits.sum(dim=1)

            if return_individuals:
                return logits, individual_logits
        else:
            logits = individual_logits
        if return_activations:
            return logits, (layer_1, layer_2, layer_3, layer_4, layer_5)
        return logits


class RedundancyNetworks(nn.Module):
    def __init__(self, n_redundant):
        super().__init__()
        self.n_redundant = n_redundant
        self.networks = [VanillaCNN() for _ in range(self.n_redundant)]
        for i, net_i in enumerate(self.networks):
            self.add_module(f'net_{i}', net_i)
        self.cached_batches = None

    def forward(self, input, return_individuals=False):
        shape = input.shape
        self.cached_batches = input[None, :].expand((self.n_redundant, shape[0], *shape[1:]))
        self.cached_batches.requires_grad_(True)
        # n_redundant can be tuned from outside
        outs = [module(b) for b, module in zip(self.cached_batches, self.networks[:self.n_redundant])]
        outs = torch.stack(outs, dim=1)
        logits = torch.sum(outs, dim=1)
        if return_individuals:
            return logits, outs
        else:
            return logits


class SmallCIFAR10GreyscaleCNN(nn.Module):
    def __init__(self, drop=0.5, num_channels=1):
        super(SmallCIFAR10GreyscaleCNN, self).__init__()

        self.num_channels = num_channels
        self.num_labels = 10

        activ = nn.ReLU(True)

        self.feature_extractor = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.num_channels, 64, 3)),
            ('relu1', activ),
            ('conv2', nn.Conv2d(64, 64, 3)),
            ('relu2', activ),
            ('maxpool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(64, 128, 3)),
            ('relu3', activ),
            ('conv4', nn.Conv2d(128, 128, 3)),
            ('relu4', activ),
            ('maxpool2', nn.MaxPool2d(2, 2)),
        ]))

        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(128 * 5 * 5, 256)),
            ('relu1', activ),
            ('drop', nn.Dropout(drop)),
            ('fc2', nn.Linear(256, 256)),
            ('relu2', activ),
            ('fc3', nn.Linear(256, self.num_labels)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.classifier.fc3.weight, 0)
        nn.init.constant_(self.classifier.fc3.bias, 0)

    def forward(self, input):
        features = self.feature_extractor(input)
        logits = self.classifier(features.view(-1, 128 * 5 * 5))
        return logits


def select_logits_on_majority_vote(logits):
    new_logits = []
    for logits_i in logits:
        argmax = torch.argmax(logits_i, dim=1).detach()
        bincount = torch.bincount(argmax)
        voted_class = torch.argmax(bincount)
        logits_to_select = argmax == voted_class
        logits_selected = logits_i[logits_to_select]
        new_logits.append(torch.sum(logits_selected, dim=0))
    new_logits = torch.stack(new_logits, dim=0)
    return new_logits


class NormalizedModel(nn.Module):
    """
    Wrapper for a model to account for the mean and std of a dataset.
    mean and std do not require grad as they should not be learned, but determined beforehand.
    mean and std should be broadcastable (see pytorch doc on broadcasting) with the data.
    Args:
        model (nn.Module): model to use to predict
        mean (torch.Tensor): sequence of means for each channel
        std (torch.Tensor): sequence of standard deviations for each channel
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super(NormalizedModel, self).__init__()

        self.model = model
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_input = (input - self.mean) / self.std
        return self.model(normalized_input)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


def requires_grad_(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad_(requires_grad)
