from config import parse_arguments

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim

from iterative_feature_removal.networks import VanillaCNN
from iterative_feature_removal import evaluate, attacks as att, train, dataloader as dl, utils as u
import numpy as np
import random
from torchvision import utils as tu


def main():
    config = parse_arguments()

    data_loaders = dl.get_data_loader(config)

    writer = SummaryWriter(config.experiment_folder)
    for arg, val in config.items():
        writer.add_text(arg, str(val), 0)

    loss_fct = u.get_loss_fct(config)

    # new model
    model = VanillaCNN().to(u.dev())
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # noise generator
    normal = torch.distributions.Normal(loc=torch.tensor(0.).to(u.dev()), scale=torch.tensor(0.5).to(u.dev()))
    uniform = torch.distributions.Uniform(low=torch.tensor(-0.5).to(u.dev()), high=torch.tensor(0.5).to(u.dev()))
    categorical = torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5]).to(u.dev()))
    noise_distributions = [normal, uniform, categorical]

    print('model loaded')
    for loop in range(config.n_loops):
        model = VanillaCNN().to(u.dev())
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        print()
        print('loop', loop)

        # train and eval
        # assert config.n_epochs > 0
        epoch_loop, accuracy_train, accuracy_test, accuracy_test_clean = 0, 0, 0, 0
        for epoch in range(config.n_epochs):
            epoch_loop = loop * config.n_epochs + epoch

            if epoch > 0:
                accuracy_train = train.train_net(config, model, optimizer, data_loaders['train'], loss_fct)
                writer.add_scalar('train/accuracy', accuracy_train, epoch_loop)

            accuracy_test = evaluate.evaluate_net(config, model, data_loaders['test'])
            accuracy_test_clean = evaluate.evaluate_net(config, model, data_loaders['clean'])
            writer.add_scalar('test/accuracy_current_dataset', accuracy_test, epoch_loop)
            writer.add_scalar('clean/accuracy', accuracy_test_clean, epoch_loop)
            print(f'i {epoch} out of ', config.n_epochs, 'acc train', accuracy_train, 'test', accuracy_test,
                  'clean', accuracy_test_clean)

        writer.add_scalar('train/final_accuracy', accuracy_train, loop)
        writer.add_scalar('test/final_accuracy_current_dataset', accuracy_test, loop)
        writer.add_scalar('clean/final_accuracy', accuracy_test_clean, loop)

        for name, data_loader in data_loaders.items():
            print('mode', name)
            new_dset, new_dset_rescaled, _ = u.get_indices_for_class_grid(data_loader.dataset.data,
                                                                          data_loader.dataset.targets,
                                                                          n_classes=config.n_classes, n_rows=8)
            new_dset = tu.make_grid(new_dset, pad_value=2, nrow=10)
            new_dset_rescaled = tu.make_grid(new_dset_rescaled, pad_value=2, nrow=10)
            writer.add_image(f'{name}_attack_{config.lp_metric}/new_dataset', new_dset, global_step=loop)
            writer.add_image(f'{name}_attack_{config.lp_metric}/new_dataset_recaled', new_dset_rescaled, global_step=loop)

            for lp, eps in zip(['l2', 'linf'], [5., 1.]):
                overwrite_dl = 'clean' not in name and lp == config.lp_metric
                print('metric', lp)
                display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, \
                linf_accuracy, success_rate, data_loaders[name] = \
                    att.evaluate_robustness(config, model, data_loader, lp_metric=lp, eps=eps,
                                            overwrite_data_loader=overwrite_dl)


                writer.add_scalar(f'{name}_attack_{lp}/l2_robustness', l2_robustness, global_step=loop)
                writer.add_scalar(f'{name}_attack_{lp}/linf_robustness', linf_robustness, global_step=loop)
                writer.add_scalar(f'{name}_attack_{lp}/l2_accuracy_eps={eps}', l2_accuracy, global_step=loop)
                writer.add_scalar(f'{name}_attack_{lp}/linf_accuracy_eps={eps}', linf_accuracy, global_step=loop)
                writer.add_image(f'{name}_attack_{lp}/adversarials', display_adv_images, global_step=loop)
                writer.add_image(f'{name}_attack_{lp}/perturbations_rescaled', display_adv_perturbations,
                                 global_step=loop)
                print()
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

