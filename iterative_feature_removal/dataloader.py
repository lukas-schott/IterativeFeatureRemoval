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
    data_loaders = {'train': data_loader_train, 'test': data_loader_test, 'clean': data_loader_test_clean}

    if config.training_mode == 'append_dataset':
        dataset_train_append = FloatTensorDataset(dataset_train.data.clone(), dataset_train.targets.clone())
        dataset_train_append.data = torch.stack([dataset_train.data, dataset_train.data], dim=1)
        data_loader_train_append = data.DataLoader(dataset_train_append, batch_size=config.batch_size, shuffle=True)
        data_loaders['train_append'] = data_loader_train_append
    return data_loaders


def create_pixel_indicator(*datasets, config=None):
    print(f'using dataset: ', config.dataset_modification)
    for dataset in datasets:
        # all_coords = np.array([[4, 4], [13, 4], [23, 4], [4, 13], [13, 13], [23, 13], [4, 23], [13, 23],
        #                    [23, 23], [20, 20]])
        all_coords = np.array([[12, 9], [14, 10], [12, 11], [14, 12], [12, 13], [14, 14], [12, 15], [14, 16],
                           [12, 17], [14, 18]])
        for i, coords in zip(range(10), all_coords):
            dataset_copy = dataset.data[dataset.targets == i]  # somehow not possible in place
            if config.dataset_modification == 'single_feat':
                dataset_copy[:] = 0
                dataset_copy = torch.clamp(dataset_copy + torch.rand(dataset_copy.shape) * 0.1, 0, 1)
                dataset_copy[:, 0, coords[0], coords[1]] = 1
                # dataset_copy[:, 0, coords[0]+1, coords[1]+1] = 1
                # dataset_copy[:, 0, coords[0]+2, coords[1]+2] = 1
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
    print('creating dataset by orthonormal projection')
    n_feats = np.prod(original_imgs.shape[1:])
    # make direction linear and unit length
    perturbations = original_imgs[is_adversarial] - perturbed_imgs[is_adversarial]
    perturbations /= torch.norm(perturbations.view((-1, n_feats)),  dim=1)[:, None, None, None] + 0.0000001

    lambdas = torch.sum(perturbations.view((-1, n_feats)) * original_imgs[is_adversarial].view(-1, n_feats), dim=1)

    new_imgs = torch.clamp(original_imgs[is_adversarial] - lambdas[:, None, None, None] * perturbations, 0, 1)
    original_imgs[is_adversarial] = new_imgs

    assert type(data_loader.dataset.data) == type(original_imgs)
    assert type(data_loader.dataset.targets) == type(original_labels)
    data_loader.dataset.data = original_imgs
    data_loader.dataset.targets = original_labels
    return data_loader


def append_dataset(config, data_loader, appended_data_loader):
    adv_imgs = data_loader.dataset.data
    rand_inds = torch.from_numpy(np.random.choice(range(adv_imgs.shape[0]),
                                                  size=int(config.percentage_to_append*adv_imgs.shape[0]),
                                                  replace=False))
    appended_data_loader.dataset.data[rand_inds, 1] = adv_imgs[rand_inds]
    return appended_data_loader


def copy_data_loader(data_loader):
    dset_new = data_loader.dataset
    dset_new.data =  data_loader.dataset.data.clone()
    dset_new.targets =  data_loader.dataset.targets.clone()
    new_data_loader = data.DataLoader(dset_new, batch_size=data_loader.batch_size, shuffle=False)
    return new_data_loader