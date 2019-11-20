import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import parse_arguments
from iterative_feature_removal import evaluate, attacks as att, dataloader as dl, utils as u
from iterative_feature_removal.networks import get_model
from iterative_feature_removal.train import get_trainer, get_optimizer
from torch.optim import lr_scheduler


def main():
    config = parse_arguments()

    data_loaders = dl.get_data_loader(config)
    if config.mnist_c:
        mnist_c_dataloaders = dl.get_mnist_c(config)

    loss_fct = u.get_loss_fct(config)
    Trainer = get_trainer(config)

    # noise generator
    # normal = torch.distributions.Normal(loc=torch.tensor(0.).to(u.dev()), scale=torch.tensor(0.5).to(u.dev()))
    # uniform = torch.distributions.Uniform(low=torch.tensor(-0.5).to(u.dev()), high=torch.tensor(0.5).to(u.dev()))
    # categorical = torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5]).to(u.dev()))
    # noise_distributions = [normal, uniform, categorical]

    model = get_model(config).to(u.dev())
    optimizer = get_optimizer(config, model.parameters())
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_decay)
    trainer = Trainer(model, data_loaders['train'], optimizer, config, loss_fct)

    writer = SummaryWriter(config.experiment_folder)
    for arg, val in config.items():
        writer.add_text(arg, str(val), 0)

    print('model initialized - now fly little electrons')
    print()

    accuracy_train, accuracy_test, accuracy_test_clean = -1, -1, -1
    best_l2 = 0
    for epoch in range(config.n_epochs):
        # redundancy
        if config.train_greedily and epoch % 10 == 0:
            print('updateing greedily')
            trainer, model, optimizer, config = u.update_for_greedy_training(trainer, model, optimizer, config, epoch,
                                                                             data_loaders, loss_fct)

        # train
        if epoch > 0 or True:
            assert id(model) == id(trainer.model)
            accuracy_train = trainer.train_epoch(epoch=epoch)
            scheduler.step(epoch)
            trainer.write_stats(writer, epoch)

        # eval
        if epoch % config.accuracy_eval_interval == 0:
            with torch.no_grad():
                accuracy_test = evaluate.evaluate_net(config, model, data_loaders['test'])
                writer.add_scalar('test/accuracy_current_dataset', accuracy_test, epoch)

                if config.dataset_modification == 'shift_mnist':
                    accuracy_test_clean = evaluate.evaluate_net(config, model, data_loaders['clean'])
                    writer.add_scalar('clean/accuracy_current_dataset', accuracy_test_clean, epoch)

                print(f'i {epoch} out of {config.n_epochs} acc train {accuracy_train:.3f} test {accuracy_test:.3f}')

        # robustness
        if epoch % config.robustness_eval_interval == 0:
            model.eval()
            adversaries = [att.get_attack(model, 'l2', attack, config.attack_eval_iter,
                                          l2_step_size=config.attack_eval_l2_step_size,
                                          linf_step_size=config.attack_eval_linf_step_size,
                                          max_eps_l2=config.attack_train_max_eps_l2,
                                          max_eps_linf=1) for attack in config.attacks_eval_names]
            # evaluate robustness
            for name, data_loader in data_loaders.items():
                if name == 'clean' and config.dataset_modification != 'shift_mnist':
                    continue
                print('eval mode', name)
                display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, \
                linf_accuracy, success_rate = \
                    att.evaluate_robustness(config, model, data_loader, adversaries)
                writer.add_scalar(f'{name}_attack_l2/l2_robustness', l2_robustness, global_step=epoch)
                writer.add_scalar(f'{name}_attack_l2/linf_robustness', linf_robustness, global_step=epoch)
                writer.add_scalar(f'{name}_attack_l2/l2_accuracy_eps={config.epsilon_threshold_accuracy_l2}',
                                  l2_accuracy, global_step=epoch)
                writer.add_scalar(f'{name}_attack_l2/linf_accuracy_eps={config.epsilon_threshold_accuracy_linf}',
                                  linf_accuracy, global_step=epoch)
                writer.add_image(f'{name}_attack_l2/adversarials', display_adv_images, global_step=epoch)
                writer.add_image(f'{name}_attack_l2/perturbations_rescaled', display_adv_perturbations,
                                 global_step=epoch)
                if name == 'test':
                    replace_best = False
                    if l2_robustness >= best_l2:
                        replace_best = True
                        print('new best l2', l2_robustness, 'will be saved as new best')
                    print('model saved')
                    u.save_state(model, optimizer, config.experiment_folder, epoch, replace_best=replace_best)
            print('eval done')

        # mnist_c
        if epoch % config.robustness_eval_interval == 0 and config.mnist_c:
            evaluate.mnist_c_evaluation(model, mnist_c_dataloaders, config, writer, epoch=epoch)


        # save
        if epoch % 20 == 0:
            u.save_state(model, optimizer, config.experiment_folder, epoch)
            print('model saved')



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

