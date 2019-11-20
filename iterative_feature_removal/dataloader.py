# from torch.utils.data import Dataset
from torchvision import datasets
import torch
from torch.utils import data
import numpy as np
from iterative_feature_removal import attacks as att
import torchvision.transforms as transforms


class FloatTensorDataset(data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.size(0)


def get_data_loader(config):
    if config.dataset == 'MNIST':
        mnist_train = datasets.MNIST(config.proj_dir + '/data/', train=True, download=True)
        mnist_test = datasets.MNIST(config.proj_dir + '/data/', train=False)
        mnist_test_clean = datasets.MNIST('./data/', train=False)
        dataset_train = FloatTensorDataset(mnist_train.data.type(torch.float32)[:, None] / 255., mnist_train.targets)
        dataset_test = FloatTensorDataset(mnist_test.data.type(torch.float32)[:, None] / 255., mnist_test.targets)
        dataset_test_clean = FloatTensorDataset(mnist_test_clean.data.type(torch.float32)[:, None] / 255.,
                                                mnist_test_clean.targets)

    elif config.dataset == 'CIFAR10' or config.dataset == 'greyscale_CIFAR10':
        # transforms
        transform_train = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]
        transform_test = [transforms.ToTensor(), RGB2Greyscale()]
        if config.dataset == 'greyscale_CIFAR10':
            transform_train.append(RGB2Greyscale())
            transform_test.append(RGB2Greyscale())
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)

        dataset_train = datasets.CIFAR10(config.proj_dir + '/data/', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10(config.proj_dir + '/data/', train=False, transform=transform_test)
        dataset_test_clean = datasets.CIFAR10(config.proj_dir + '/data/', train=False, transform=transform_test)

    end = config.end
    dataset_train.data = dataset_train.data[:end]
    dataset_train.targets = dataset_train.targets[:end]
    dataset_test.data = dataset_test.data[:end]
    dataset_test.targets = dataset_test.targets[:end]
    dataset_test_clean.data = dataset_test_clean.data[:end]
    dataset_test_clean.targets = dataset_test_clean.targets[:end]

    if config.dataset_modification != 'None':
        dataset_train, dataset_test =  create_pixel_indicator(dataset_train, dataset_test, config=config)

    data_loader_train = data.DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    data_loader_test = data.DataLoader(dataset_test, batch_size=config.batch_size, shuffle=False)
    data_loader_test_clean = data.DataLoader(dataset_test_clean, batch_size=config.batch_size, shuffle=False)
    data_loaders = {'train': data_loader_train, 'test': data_loader_test, 'clean': data_loader_test_clean}

    return data_loaders


def get_mnist_c(config):
    n_worker = 4
    data_loaders_names = \
        {'Brightness': 'brightness',
         'Canny Edges': 'canny_edges',
         'Dotted Line': 'dotted_line',
         'Fog': 'fog',
         'Glass Blur': 'glass_blur',
         'Identity': 'identity',
         'Impulse Noise': 'impulse_noise',
         'Motion Blur': 'motion_blur',
         'Rotate': 'rotate',
         'Scale': 'scale',
         'Shear': 'shear',
         'Shot Noise': 'shot_noise',
         'Spatter': 'spatter',
         'Stripe': 'stripe',
         'Translate': 'translate',
         'Zigzag': 'zigzag'}
    data_loaders = {}
    for name, path in data_loaders_names.items():
        data_loaders[name] = {}

        test_images = np.load(config.proj_dir + '/data/mnist_c/' + path + '/test_images.npy')
        test_labels = np.load(config.proj_dir + '/data/mnist_c/' + path + '/test_labels.npy')

        test_images = test_images.squeeze(-1)
        test_images = np.expand_dims(test_images, axis=1)

        tensor_x = torch.from_numpy(test_images)  # transform to torch tensors
        tensor_y = torch.from_numpy(test_labels)

        custom_dataset = data.TensorDataset(tensor_x, tensor_y)  # create your datset
        data_loaders[name] = data.DataLoader(custom_dataset, batch_size=config.batch_size_test,
                                             num_workers=n_worker)
    return data_loaders


def create_pixel_indicator(*datasets, config=None):
    print(f'using dataset: ', config.dataset_modification)
    for dataset in datasets:
        all_coords = np.array([[4, 4], [13, 4], [23, 4], [4, 13], [13, 13], [23, 13], [4, 23], [13, 23],
                           [23, 23], [20, 20]])
        # all_coords = np.array([[12, 9], [14, 10], [12, 11], [14, 12], [12, 13], [14, 14], [12, 15], [14, 16],
        #                    [12, 17], [14, 18]])
        for i, coords in zip(range(10), all_coords):
            dataset_copy = dataset.data[dataset.targets == i]  # somehow not possible in place
            if config.dataset_modification == 'single_feat':
                dataset_copy[:] = 0
                dataset_copy = torch.clamp(dataset_copy + torch.rand(dataset_copy.shape) * 0.1, 0, 1)
                dataset_copy[:, 0, coords[0], coords[1]] = 1
                # dataset_copy[:, 0, coords[0]+1, coords[1]+1] = 1
                # dataset_copy[:, 0, coords[0]+2, coords[1]+2] = 1
            elif config.dataset_modification == 'shift_mnist':
                print('shift mnists', i)
                dataset_copy[:, 0, i*2+4, 4] = 1
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
    original_imgs[is_adversarial] = att.orthogonal_projection(original_imgs[is_adversarial],
                                                              perturbed_imgs[is_adversarial])
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


def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.333, dtype=dtype), np.asarray(.333, dtype=dtype), np.asarray(.333, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst


class RGB2Greyscale(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, factors=(0.333, 0.333, 0.333)):
        self.factors = factors

    def __call__(self, img):
        grey_img = self.factors[0] * img[0] + self.factors[1] * img[1] + self.factors[2] * img[2]
        return grey_img[None]

    def __repr__(self):
        return self.__class__.__name__