# from torch.utils.data import Dataset
from torchvision import datasets
import torch
from torch.utils import data
import numpy as np


class FloatTensorDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.size(0)


def get_data_loader(config):
    mnist_train = datasets.MNIST(config.proj_dir + '/data/', train=True, download=True)
    mnist_test = datasets.MNIST(config.proj_dir + '/data/', train=False)
    mnist_test_clean = datasets.MNIST('./data/', train=False)
    dataset_train = FloatTensorDataset(mnist_train.data.type(torch.float32) / 255., mnist_train.targets)
    dataset_test = FloatTensorDataset(mnist_test.data.type(torch.float32) / 255., mnist_test.targets)
    dataset_test_clean = FloatTensorDataset(mnist_test_clean.data.type(torch.float32) / 255., mnist_test_clean.targets)

    end = config.end
    dataset_train.data = dataset_train.data[:end, None]
    dataset_train.targets = dataset_train.targets[:end]
    dataset_test.data = dataset_test.data[:end, None]
    dataset_test.targets = dataset_test.targets[:end]
    dataset_test_clean.data = dataset_test_clean.data[:end, None]
    dataset_test_clean.targets = dataset_test_clean.targets[:end]

    if config.dataset_modification != 'None':
        dataset_train, dataset_test, dataset_test_clean = \
            create_pixel_indicator(dataset_train, dataset_test, dataset_test_clean, config=config)

    print('dataset', torch.min(dataset_train.data), torch.max(dataset_train.data))
    data_loader_train = data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    data_loader_test_clean = data.DataLoader(dataset_test_clean, batch_size=config.batch_size, shuffle=False)
    return {'train': data_loader_train, 'test': data_loader_test, 'clean': data_loader_test_clean}


def create_pixel_indicator(*datasets, config=None):
    print(f'using dataset: ', config.dataset_modification)
    for dataset in datasets:
        for i in range(10):
            dataset_copy = dataset.data[dataset.targets == i]  # somehow not possible in place
            if config.dataset_modification == 'single_feat':
                # dataset_copy[:] = 0
                # dataset_copy[:, 0, i, 0] = 1

                if i == 0:
                    dataset_copy[:, 0, 4, 4] = 1
                if i == 1:
                    dataset_copy[:, 0, 13, 4] = 1
                if i == 2:
                    dataset_copy[:, 0, 23, 4] = 1
                if i == 3:
                    dataset_copy[:, 0, 4, 13] = 1
                if i == 4:
                    dataset_copy[:, 0, 13, 13] = 1
                if i == 5:
                    dataset_copy[:, 0, 23, 13] = 1
                if i == 6:
                    dataset_copy[:, 0, 4, 23] = 1
                if i == 7:
                    dataset_copy[:, 0, 13, 23] = 1
                if i == 8:
                    dataset_copy[:, 0, 23, 23] = 1
                if i == 9:
                    dataset_copy[:, 0, 20, 20] = 1

                dataset_copy = torch.clamp(dataset_copy + torch.rand(dataset_copy.shape) * 0.1, 0, 1)
            elif config.dataset_modification == 'shift_mnist':
                dataset_copy[:, 0, i, 0] = 1
            elif config.dataset_modification == 'double_feat':
                dataset_copy[:] = 0
                dataset_copy[:, 0, i, 0] = 1
                dataset_copy[:, 0, i, 2] = 1
                dataset_copy = torch.clamp(dataset_copy + torch.rand(dataset_copy.shape) * 0.1, 0, 1)
            else:
                raise Exception(f'dataset {config.dataset_modification} not selectable')
            dataset.data[dataset.targets == i] = dataset_copy
    return datasets


def create_new_dataset(config, perturbed_imgs, original_imgs, original_labels,
                       is_adversarial, data_loader):
    print('creating dataset')
    n_feats = np.prod(original_imgs.shape[1:])
    perturbations = original_imgs[is_adversarial] - perturbed_imgs[is_adversarial]
    perturbations /= torch.norm(perturbations.view((-1, n_feats)),  dim=1)[:, None, None, None] + 0.0000001
    lambdas = torch.sum(perturbations.view((-1, n_feats)) * original_imgs[is_adversarial].view(-1, n_feats), dim=1)
    original_imgs[is_adversarial] = \
        torch.clamp(original_imgs[is_adversarial] - lambdas[:, None, None, None] * perturbations, 0, 1)

    assert type(data_loader.dataset.data) == type(original_imgs)
    assert type(data_loader.dataset.targets) == type(original_labels)
    data_loader.dataset.data = original_imgs
    data_loader.dataset.targets = original_labels
    return data_loader
