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

            if epoch % config.eval_interval == 0:

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

