from config import parse_arguments

import torch
from torch.utils.tensorboard import SummaryWriter
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

    # new model
    model = VanillaCNN().to(u.dev())
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # noise generator
    normal = torch.distributions.Normal(loc=torch.tensor(0.).to(u.dev()), scale=torch.tensor(0.5).to(u.dev()))
    uniform = torch.distributions.Uniform(low=torch.tensor(-0.5).to(u.dev()), high=torch.tensor(0.5).to(u.dev()))
    categorical = torch.distributions.Categorical(probs=torch.tensor([0.5, 0.5]).to(u.dev()))
    noise_distributions = [normal, uniform, categorical]

    print('model loaded, starting training')
    last_linf_accuracy = 0
    for epoch in range(config.n_epochs):

        # adv attack
        for data_loader, name in zip([data_loader_train, data_loader_test], ['train', 'test']):
            model.eval()
            accuracy_clean = evaluate.evaluate_net(config, model, data_loader, adv_training=False)
            writer.add_scalar(f'{name}/clean_accuracy', accuracy_clean, global_step=epoch)

            # measure noise robustness
            for noise_distribution in noise_distributions:
                accuracy_noise, imgs_noise = att.measure_noise_robustness(config, model, data_loader,
                                                                          noise_distribution)

                writer.add_scalar(f'{name}_noise/{noise_distribution.__str__()}', accuracy_noise, global_step=epoch)
                writer.add_image(f'{name}_noise/{noise_distribution.__str__()}', imgs_noise, global_step=epoch)

            # measure adversarial robustness
            for lp_metric, eps in zip(['l2', 'linf'], [10., 1.]):
                display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, linf_accuracy,  success_rate = \
                    att.evaluate_robustness(config, model, data_loader, lp_metric=lp_metric, eps=eps, train=False)

                writer.add_scalar(f'{name}_attack_{lp_metric}/l2_robustness', l2_robustness, global_step=epoch)
                writer.add_scalar(f'{name}_attack_{lp_metric}/l2_accuracy_eps=1.5', l2_accuracy, global_step=epoch)
                writer.add_scalar(f'{name}_attack_{lp_metric}/linf_robustness', linf_robustness, global_step=epoch)
                writer.add_scalar(f'{name}_attack_{lp_metric}/linf_accuracy_eps=0.3', linf_accuracy, global_step=epoch)
                writer.add_image(f'{name}_attack_{lp_metric}/adversarials', display_adv_images, global_step=epoch)
                writer.add_image(f'{name}_attack_{lp_metric}/perturbations_rescaled', display_adv_perturbations,
                                 global_step=epoch)

        # save network
        torch.save(model.state_dict(), config.experiment_folder + '/save_model_most_recent.pt')
        torch.save(optimizer.state_dict(), config.experiment_folder + '/save_optimizer_most_recent.pt')
        if last_linf_accuracy <= linf_accuracy:
            torch.save(model.state_dict(), config.experiment_folder + '/save_model_best.pt')
            torch.save(optimizer.state_dict(), config.experiment_folder + '/save_optimizer_best.pt')

        # train and eval
        accuracy_adv_train = train.train_net(config, model, optimizer, data_loader_train, loss_fct,
                                             adv_training=config.adv_training)
        writer.add_scalar(f'train/accuracy_perturbed_init', accuracy_adv_train, epoch)

        print(f'epoch {epoch} out of ', config.n_epochs, 'clean test', accuracy_clean, 'adv acc train', accuracy_adv_train)

    writer.close()


if __name__ == '__main__':
    import traceback
    from colorama import Fore, init

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # for replicable results this can be activated
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    init()

    try:
        main()
    except Exception:
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
