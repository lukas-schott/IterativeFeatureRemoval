from config import parse_arguments

import numpy as np


a = np.zeros((10, 5))
b = a[:10000]
print('b', b.shape)


config = parse_arguments()

print('n epoch', config.n_epochs)


print(config.__dict__)
print(dir(config))
print(vars(config))
o = config

keys = [f for f in dir(o) if not callable(getattr(o, f)) and not f.startswith('__')]
new_dict = {}
for key in keys:

    new_dict[key] = config.__dict__[key]

print('new dict', new_dict)

# dset_train = datasets.MNIST('./data/MNIST/', train=True, download=True,
#                             transform=transforms.ToTensor())
# dset_test = datasets.MNIST('./data/MNIST/', train=False,
#                            transform=transforms.ToTensor())
#
# dataloader_train = data.DataLoader(dset_train, batch_size=config.batch_size)
#
# print('here')
# print(type(dataloader_train.dataset.data), dataloader_train.dataset.data.shape)