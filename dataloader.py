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
    mnist_train = datasets.MNIST('./data/MNIST/', train=True, download=True)
    mnist_test = datasets.MNIST('./data/MNIST/', train=False)
    mnist_test_clean = datasets.MNIST('./data/MNIST/', train=False)
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

    if config.shift_mnist or config.single_feat:
        dataset_train, dataset_test, dataset_test_clean = \
            create_pixel_indicator(dataset_train, dataset_test, dataset_test_clean, config=config)

    print('dataset', torch.min(dataset_train.data), torch.max(dataset_train.data))
    data_loader_train = data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    data_loader_test_clean = data.DataLoader(dataset_test_clean, batch_size=config.batch_size, shuffle=False)
    return data_loader_train, data_loader_test, data_loader_test_clean


def create_pixel_indicator(*datasets, config=None):
    for dataset in datasets:
        for i in range(10):
            dataset_copy = dataset.data[dataset.targets == i]  # somehow not possible in place
            if config.single_feat:
                print('single feat dataset')
                dataset_copy[:] = 0
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
            else:
                print('using shift mnist')
                dataset_copy[:, 0, i, 0] = 1
            dataset.data[dataset.targets == i] = dataset_copy
    return datasets


def create_new_dataset(config, perturbed_imgs, original_imgs, original_labels,
                       is_adversarial, data_loader):
    print('creating dataset')
    n_feats = np.prod(original_imgs.shape[1:])
    perturbations = original_imgs[is_adversarial] - perturbed_imgs[is_adversarial]
    perturbations /= np.linalg.norm(perturbations.reshape((-1, n_feats)),  axis=1)[:, None, None, None] + 0.0000001
    lambdas = np.sum(perturbations.reshape((-1, n_feats)) * original_imgs[is_adversarial].reshape(-1, n_feats), axis=1)
    original_imgs[is_adversarial] = \
        np.clip(original_imgs[is_adversarial] - lambdas[:, None, None, None] * perturbations, 0, 1)

    mask = np.empty(original_imgs.shape[0], dtype=np.bool)
    mask[:] = True
    if config.er != 0:
        mask = np.random.choice([True, False], original_imgs.shape[0], p=[1 - config.er, config.er])

    data_loader.dataset.data[mask] = torch.from_numpy(original_imgs)[mask]
    data_loader.dataset.targets[mask] = torch.from_numpy(original_labels)[mask]
