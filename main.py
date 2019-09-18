from config import parse_arguments

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tu
import dataloader as dl
from torch import optim

import attacks as att
from networks import VanillaCNN
import utils as u
import train
import evaluate
import numpy as np
import random


def main():
    config = parse_arguments()

    data_loader_train, data_loader_test, data_loader_test_clean = dl.get_data_loader(config)

    writer = SummaryWriter(config.experiment_folder)
    for arg, val in config.items():
        writer.add_text(arg, str(val), 0)

    loss_fct = u.get_loss_fct(config)

    print('model loaded')
    for loop in range(config.n_loops):
        print()
        print('loop', loop)

        # new model
        model = VanillaCNN().to(u.dev())
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # plot datasets
        for name, data_loader in zip(['train', 'test', 'clean'],
                                     [data_loader_train, data_loader_test, data_loader_test_clean]):

            images, images_rescaled = u.get_indices_for_class_grid(data_loader.dataset.data, data_loader.dataset.targets,
                                                                   n_classes=config.n_classes, n_rows=8)
            images_grid = tu.make_grid(images, pad_value=2, nrow=10)
            images_rescaled_grid = tu.make_grid(images_rescaled, pad_value=2, nrow=10)
            writer.add_image(f'{name}/modified_images', images_grid, global_step=loop)
            writer.add_image(f'{name}/modified_images_rescaled', images_rescaled_grid, global_step=loop)
            writer.add_scalar(f'{name}/dataset_magnitude', torch.mean(data_loader.dataset.data ** 2), global_step=loop)

        # train and eval
        assert config.n_epochs > 0
        epoch_loop, accuracy_train, accuracy_test, accuracy_test_clean = 0, 0, 0, 0
        for epoch in range(config.n_epochs):
            epoch_loop = loop * config.n_epochs + epoch

            if epoch > 0:
                accuracy_train = train.train_net(config, model, optimizer, data_loader_train, loss_fct)
                writer.add_scalar('train/accuracy', accuracy_train, epoch_loop)

            accuracy_test = evaluate.evaluate_net(config, model, data_loader_test)
            accuracy_test_clean = evaluate.evaluate_net(config, model, data_loader_test_clean)
            writer.add_scalar('test/accuracy_current_dataset', accuracy_test, epoch_loop)
            writer.add_scalar('clean/accuracy', accuracy_test_clean, epoch_loop)
            print(f'i {epoch} out of ', config.n_epochs, 'acc train', accuracy_train, 'test', accuracy_test,
                  'clean', accuracy_test_clean)

        writer.add_scalar('train/final_accuracy', accuracy_train, loop)
        writer.add_scalar('test/final_accuracy_current_dataset', accuracy_test, loop)
        writer.add_scalar('clean/final_accuracy', accuracy_test_clean, loop)

        # adv attack and create new dataset
        for data_loader, name in zip([data_loader_train, data_loader_test, data_loader_test_clean],
                                     ['train', 'test', 'clean']):
            display_adv_images, display_adv_perturbations, l2_robustness, success_rate = \
                att.create_adversarial_dataset(config, model, data_loader, keep_data_loader='clean' in name)

            writer.add_image(f'{name}/perturbations_rescaled', display_adv_perturbations, global_step=loop)
            writer.add_image(f'{name}/adversarials', display_adv_images, global_step=loop)
            writer.add_scalar(f'{name}/l2_robustness', l2_robustness, global_step=loop)
            writer.add_scalar(f'{name}/attack_success_rate', success_rate, global_step=loop)

    writer.close()


if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for replicable results this can be activated
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()

