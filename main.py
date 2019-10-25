import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as tu

from config import parse_arguments
from iterative_feature_removal import evaluate, attacks as att, dataloader as dl, utils as u
from iterative_feature_removal.networks import get_model
from iterative_feature_removal.train import get_trainer, get_optimizer
from torch.optim import lr_scheduler


def main():
    config = parse_arguments()

    data_loaders = dl.get_data_loader(config)

    writer = SummaryWriter(config.experiment_folder)
    for arg, val in config.items():
        writer.add_text(arg, str(val), 0)

    loss_fct = u.get_loss_fct(config)
    Trainer = get_trainer(config)

    # noise generator
    # normal = torch.distributions.Normal(loc=torch.tensor(0.).to(u.dev()), scale=torch.tensor(0.5).to(u.dev()))
    # uniform = torch.distributions.Uniform(low=torch.tensor(-0.5).to(u.dev()), high=torch.tensor(0.5).to(u.dev()))
    # categorical = torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5]).to(u.dev()))
    # noise_distributions = [normal, uniform, categorical]

    print('model loaded - now fly little electrons')
    for loop in range(config.n_loops):
        if loop == 0 or config.reinit_network:
            model = get_model(config).to(u.dev())
            optimizer = get_optimizer(config, model)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_decay)
            trainer = Trainer(model, data_loaders['train'], optimizer, config, loss_fct)
        print()
        print('loop', loop)

        # train and eval
        # assert config.n_epochs > 0
        epoch_loop, accuracy_train, accuracy_test, accuracy_test_clean = 0, 0, 0, 0
        for epoch in range(config.n_epochs):
            epoch_loop = loop * config.n_epochs + epoch
            if epoch > 0:
                assert id(model) == id(trainer.model)
                accuracy_train = trainer.train_epoch()
                scheduler.step(epoch_loop)
                writer.add_scalar('train/accuracy', accuracy_train, epoch_loop)
            accuracy_test = evaluate.evaluate_net(config, model, data_loaders['test'])
            accuracy_test_clean = evaluate.evaluate_net(config, model, data_loaders['clean'])
            writer.add_scalar('test/accuracy_current_dataset', accuracy_test, epoch_loop)
            writer.add_scalar('clean/accuracy', accuracy_test_clean, epoch_loop)
            print(f'i {epoch} out of {config.n_epochs} acc train {accuracy_train:.3f} test {accuracy_test:.3f}'
                  f' clean {accuracy_test_clean:.3f}')
        writer.add_scalar('train/final_accuracy', accuracy_train, loop)
        writer.add_scalar('test/final_accuracy_current_dataset', accuracy_test, loop)
        writer.add_scalar('clean/final_accuracy', accuracy_test_clean, loop)

        model.eval()
        adversaries = [att.get_attack(model, 'l2', attack, config.attack_iter,
                                      l2_step_size=config.attack_l2_step_size,
                                      linf_step_size=config.attack_linf_step_size,
                                      max_eps_l2=config.epsilon_max_l2,
                                      max_eps_linf=1) for attack in config.attacks_names]
        # evaluate robustness
        for name, data_loader in data_loaders.items():
            print('eval mode', name)
            if 'append' in name:
                continue
            display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, \
            linf_accuracy, success_rate, data_loaders[name] = \
                att.evaluate_robustness(config, model, data_loader, adversaries)
            writer.add_scalar(f'{name}_attack_l2/l2_robustness', l2_robustness, global_step=loop)
            writer.add_scalar(f'{name}_attack_l2/linf_robustness', linf_robustness, global_step=loop)
            writer.add_scalar(f'{name}_attack_l2/l2_accuracy_eps={1.5}', l2_accuracy, global_step=loop)
            writer.add_scalar(f'{name}_attack_l2/linf_accuracy_eps={0.3}', linf_accuracy, global_step=loop)
            writer.add_image(f'{name}_attack_l2/adversarials', display_adv_images, global_step=loop)
            writer.add_image(f'{name}_attack_l2/perturbations_rescaled', display_adv_perturbations,
                             global_step=loop)

        # # overwrite dataset
        # if config.training_mode == 'project_out':
        #     print('project out features, overwrite training dataset')
        #     for name in ['train', 'test']:
        #         data_loader = data_loaders[name]
        #         adversary = att.get_attack(model, 'l2', config.attack_for_new_dataset, config.attack_iter,
        #                                    l2_step_size=config.attack_l2_step_size,
        #                                    linf_step_size=config.attack_linf_step_size,
        #                                    max_eps_l2=config.max_eps_new_dataset,
        #                                    max_eps_linf=1)
        #         data_loaders['train'] = att.generate_new_dataset(config, model, data_loaders['train'], adversary)
        #         data_loaders['test'] = att.generate_new_dataset(config, model, data_loaders['test'], adversary)
        #
        #         new_dset, new_dset_rescaled, _ = u.get_indices_for_class_grid(data_loader.dataset.data,
        #                                                                       data_loader.dataset.targets,
        #                                                                       n_classes=config.n_classes, n_rows=8)
        #         new_dset = tu.make_grid(new_dset, pad_value=2, nrow=10)
        #         new_dset_rescaled = tu.make_grid(new_dset_rescaled, pad_value=2, nrow=10)
        #         writer.add_image(f'{name}_attack_{config.lp_metric}/new_dataset', new_dset, global_step=loop+1)
        #         writer.add_image(f'{name}_attack_{config.lp_metric}/new_dataset_recaled', new_dset_rescaled,
        #                          global_step=loop+1)
        #
        # if config.training_mode == 'append_dataset':
        #     adversary = att.get_attack(model, 'l2', config.attack_for_new_dataset, config.attack_iter,
        #                                l2_step_size=config.attack_l2_step_size,
        #                                linf_step_size=config.attack_linf_step_size,
        #                                max_eps_l2=config.max_eps_new_dataset,
        #                                max_eps_linf=1)
        #     data_loader_adv = dl.copy_data_loader(data_loaders['train'])
        #     data_loader_adv = att.generate_new_dataset(config, model, data_loader_adv, adversary)
        #     data_loaders['train_append'] = dl.append_dataset(config,
        #                                                      data_loader_adv,
        #                                                      data_loaders['train_append'])
        #
        #     imgs = tu.make_grid(
        #         data_loaders['train_append'].dataset.data.reshape(
        #             data_loaders['train_append'].dataset.data.shape[0] * 2,
        #             1, 28, 28)[:20], pad_value=2, nrow=10)
        #     writer.add_image(f'apped/new_dataset', imgs, global_step=loop + 1)

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

