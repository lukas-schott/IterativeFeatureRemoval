import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import parse_arguments, c2d
from iterative_feature_removal import evaluate
from iterative_feature_removal import attacks as att, dataloader as dl, utils as u
from iterative_feature_removal.networks import get_model, Subnetwork
from iterative_feature_removal.train import get_trainer, get_optimizer
from torch.optim import lr_scheduler


def main():
    config = parse_arguments()

    # data
    data_loaders = dl.get_data_loader(config)
    data_loaders_robustness = [('test', data_loaders['test'])]
    if config.dataset_modification == 'shift_mnist' or config.dataset_modification == 'texture_mnist':
        data_loaders_robustness.append(('clean', data_loaders['clean']))

    if config.mnist_c:
        mnist_c_dataloaders = dl.get_mnist_c(config)

    loss_fct = u.get_loss_fct(config)
    trainer = get_trainer(config)

    model = get_model(config).to(u.dev())
    optimizer = get_optimizer(config, model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_decay)
    trainer = trainer(model, data_loaders['train'], optimizer, config, loss_fct)

    writer = SummaryWriter(config.experiment_folder)
    for arg, val in config.items():
        writer.add_text(arg, str(val), 0)

    print('model initialized - now fly little electrons')
    print()
    if config.all_in_one_model and config.train_greedily:
        trainer.mask = torch.ones(config.n_redundant).type(torch.bool).to(u.dev())

    accuracy_train, accuracy_test, accuracy_test_clean = -1, -1, -1
    best_l2 = 0
    last_epoch = False
    j = 0
    for epoch in range(config.n_epochs):
        if epoch == config.n_epochs - 1:
            last_epoch = True

        # redundancy
        if config.train_greedily and epoch % config.n_epochs_per_net == 0 and not config.all_in_one_model:
            print('updating greedily')
            trainer, model, optimizer, config = u.update_for_greedy_training(trainer, model, optimizer, config, epoch,
                                                                             data_loaders, loss_fct)
        if config.all_in_one_model and epoch % config.n_epochs_per_net == 0 and config.train_greedily:
            print('upating trainer')
            j += 1
            trainer.mask[:j] = False
            print('trainer mask', trainer.mask)

        if epoch > 0 or config.debug:
            print('train')
            assert id(model) == id(trainer.model)
            accuracy_train = trainer.train_epoch(epoch=epoch)
            scheduler.step(epoch)
            trainer.write_stats(writer, epoch)

        # eval
        if epoch % config.accuracy_eval_interval == 0 or last_epoch:
            with torch.no_grad():
                accuracy_test = evaluate.evaluate_net(config, model, data_loaders['test'])
                writer.add_scalar('test/accuracy_current_dataset', accuracy_test, epoch)

                clean_acc_str = ''
                if config.dataset_modification == 'shift_mnist' or config.dataset_modification == 'texture_mnist':
                    accuracy_test_clean = evaluate.evaluate_net(config, model, data_loaders['clean'])
                    writer.add_scalar('clean/accuracy_current_dataset', accuracy_test_clean, epoch)
                    clean_acc_str = f'clean accuracy {accuracy_test_clean:.3f}'

                print(f'i {epoch} out of {config.n_epochs} acc train {accuracy_train:.3f} test {accuracy_test:.3f} '
                      + clean_acc_str)

        # test cosine similarities
        if epoch % config.accuracy_eval_interval == 0 and model.n_redundant > 0 or last_epoch:
            if model.n_redundant > 0:
                evaluate.plot_similarities(config, model, data_loaders['test'], writer, epoch)

        # robustness
        if epoch % config.robustness_eval_interval == 0 or last_epoch:

            models = [('model', model)]
            if model.n_redundant > 1 and config.training_mode == 'redundancy':
                if config.all_in_one_model:
                    models += [(f'model_{i}', Subnetwork(model, i)) for i in range(model.n_redundant)]
                else:
                    models += [(f'model_{i}', net_i) for i, net_i in enumerate(model.networks[:model.n_redundant])]

            for dl_name, data_loader in data_loaders_robustness:
                all_perturbed_data = []
                for m_name, eval_model in models:
                    name = f'{dl_name}_{m_name}'
                    eval_model.eval()

                    # adversarial robustness
                    adversaries = [att.get_attack(eval_model, 'l2', attack, config.attack_eval_iter,
                                                  l2_step_size=config.attack_eval_l2_step_size,
                                                  linf_step_size=config.attack_eval_linf_step_size,
                                                  max_eps_l2=config.attack_train_max_eps_l2,
                                                  max_eps_linf=1) for attack in config.attacks_eval_names.split(', ')]
                    print('robustness eval mode', name)
                    display_imgs, l2_robustness, l2_accuracy, _, _, success_rate, perturbed_data = \
                        att.evaluate_robustness(config, eval_model, data_loader, adversaries)
                    writer.add_scalar(f'{name}_attack_l2/l2_robustness', l2_robustness, global_step=epoch)
                    writer.add_scalar(f'{name}_attack_l2/l2_accuracy_eps={config.epsilon_threshold_accuracy_l2}',
                                      l2_accuracy, global_step=epoch)
                    writer.add_image(f'{name}_attack_l2/adversarials', display_imgs['adversarials'], global_step=epoch)
                    writer.add_image(f'{name}_attack_l2/originals', display_imgs['originals'], global_step=epoch)
                    writer.add_image(f'{name}_attack_l2/gradients', display_imgs['gradients'], global_step=epoch)
                    writer.add_image(f'{name}_attack_l2/perturbations_rescaled', display_imgs['perturbations_rescaled'],
                                     global_step=epoch)
                    writer.add_image(f'{name}_attack_l2/gradients_adv_direction',
                                     display_imgs['gradients_adv_direction'], global_step=epoch)

                    accuracy_plane = evaluate.evaluate_net(config, eval_model, data_loader)
                    writer.add_scalar(f'individuals_{dl_name}/{m_name}_accuracy', accuracy_plane, epoch)

                    # shift and rotation
                    accuracy_shift_rot = att.test_under_shift_rotation(config, eval_model, data_loaders['test'])
                    writer.add_scalar(f'{name}_shift_and_rotation/accuracy', accuracy_shift_rot, global_step=epoch)

                    all_perturbed_data.append(perturbed_data)

                # transferability
                assert len(models) == len(all_perturbed_data)
                transferability_matrix = torch.zeros((len(models), len(models)))
                for m, (_, eval_model) in enumerate(models):
                    for p, perturbed_data in enumerate(all_perturbed_data):
                        acc_perturbed = evaluate.evaluate_net(config, eval_model, [perturbed_data])
                        transferability_matrix[p, m] = 1. - acc_perturbed
                fig = u.plot_similarity_matrix(transferability_matrix, names=[name for name, _ in models],
                                               ylabel='source', xlabel='target')

                writer.add_figure(f'{dl_name}_adversarial_transferability/', fig, epoch)

            # if name == 'test':
                #     replace_best = False
                #     if l2_robustness >= best_l2:
                #         replace_best = True
                #         print('new best l2', l2_robustness, 'will be saved as new best')
                #     print('model saved')
                #     u.save_state(model, optimizer, config.experiment_folder, epoch, replace_best=replace_best)
            print('eval done')

        # black box robustness
        if (epoch % config.black_box_attack_interval == 0 or last_epoch) and epoch > 0 and not config.debug:
            print('running boundary attack with ', config.boundary_attack_iter)
            l2_robustness, adversarials = att.run_boundary_attack(model, data_loaders['test'],
                                                                  n_iter=config.boundary_attack_iter)
            writer.add_scalar(f'test_boundary_attack_l2/l2_robustness', l2_robustness, global_step=epoch)
            writer.add_image(f'test_attack_l2/boundary_attack_adversarials', adversarials, global_step=epoch)

        # mnist_c
        if epoch % config.robustness_eval_interval == 0 and config.mnist_c or last_epoch:
            evaluate.mnist_c_evaluation(model, mnist_c_dataloaders, config, writer, epoch=epoch)

        # save
        if epoch % 20 == 0 or epoch == config.n_epochs - 1 or last_epoch:
            u.save_state(model, optimizer, config.experiment_folder, epoch, config)
            print('model saved')

    writer.add_hparams(c2d(config), metric_dict={"hparam/test_accuracy": 0.})
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

