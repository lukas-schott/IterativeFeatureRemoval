import foolbox
import torch
from advertorch import attacks as pyatt
from torch.utils import data
from torchvision import utils as tu

import utils as u

print('foolbox', foolbox.__version__, foolbox.__path__)


def measure_noise_robustness(config, model, data_loader, noise_distribution):
    bs = 10000
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=bs, shuffle=False)

    n_correct, n_total = 0., 0.
    for i, (b, l) in enumerate(sorted_data_loader):
        b, l = b.cuda(), l.cuda()
        noise = noise_distribution.sample(sample_shape=b.shape).type(torch.float32)
        b = torch.clamp(b + noise, 0, 1)
        pred = torch.argmax(model(b), dim=1)
        n_correct += torch.sum(pred == l)
        n_total += b.shape[0]
        if i == 0:
            l_plot = l
            b_plot = b

    accuracy = float(n_correct) / n_total

    display_noise_images, _, _ = u.get_indices_for_class_grid(b_plot, l_plot, n_classes=config.n_classes, n_rows=5)
    display_noise_images = tu.make_grid(display_noise_images, pad_value=2, nrow=5)

    return accuracy, display_noise_images


def calculate_adversarial_perturbations_batch(config, model, b, l, rand_init):
    adversary = get_attack(model, lp_metric=config.lp_metric, eps=config.adv_epsilon, rand_init=rand_init)
    adv_untargeted = adversary.perturb(b, l)
    return adv_untargeted


def attack_batch(config, model, b, l):
    adv_imgs = calculate_adversarial_perturbations_batch(config, model, b, l)
    perturbations = adv_imgs - b
    is_adversarial = [(torch.argmax(model(adv_imgs), dim=1) != l).cpu().type(torch.bool)]
    adv_imgs = torch.zeros_like(adv_imgs)
    adv_imgs[is_adversarial] = adv_imgs[is_adversarial]
    return adv_imgs, perturbations


def get_attack(model, lp_metric, eps, rand_init):
    if lp_metric == 'l2':
        adversary = pyatt.L2PGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                                      targeted=False, rand_init=rand_init)
    elif lp_metric == 'linf':
        adversary = pyatt.LinfPGDAttack(model, loss_fn=torch.nn.CrossEntropyLoss(reduction="sum"), eps=eps,
                                        targeted=False, rand_init=rand_init)
    else:
        raise Exception(f'lp metric {lp_metric} is not defined.')
    return adversary


def get_adversarial_perturbations(config, model, data_loader, lp_metric, eps, rand_init=True):
    print('running adv attack')
    print()
    model.eval()
    adversary = get_attack(model, lp_metric, eps, rand_init)

    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=10000, shuffle=False)
    adversarial_images = []
    is_adversarial = []
    original_images = []
    original_targets = []
    for i, (b, l) in enumerate(sorted_data_loader):
        b, l = b.to(u.dev()), l.to(u.dev())
        u.check_bounds(b)
        print(f'i {i} out of ', len(sorted_data_loader))

        adversarials = adversary.perturb(b, l).detach()
        with torch.no_grad():
            is_adversarial += [(torch.argmax(model(adversarials), dim=1) != l).cpu().type(torch.bool)]
        adversarial_images += [adversarials.cpu()]

        original_images += [b.cpu()]
        original_targets += [l.cpu()]
        break
    perturbed_images = torch.cat(adversarial_images, dim=0)
    is_adversarial = torch.cat(is_adversarial, dim=0)  # ignore it if attack failed
    original_images = torch.cat(original_images, dim=0)
    original_targets = torch.cat(original_targets, dim=0)
    return perturbed_images, original_images, original_targets, is_adversarial


def evaluate_robustness(config, model, data_loader, lp_metric, eps, rand_init):
    perturbed_images, original_images, original_targets, is_adversarial = \
        get_adversarial_perturbations(config, model, data_loader, lp_metric, eps=eps, rand_init=rand_init)
    adv_perturbations = perturbed_images - original_images
    _, display_adv_perturbations, _ = u.get_indices_for_class_grid(adv_perturbations[is_adversarial],
                                                                   original_targets[is_adversarial],
                                                                   n_classes=config.n_classes, n_rows=8)
    display_adv_perturbations = tu.make_grid(display_adv_perturbations, pad_value=2, nrow=10)

    display_adv_images, _, _ = u.get_indices_for_class_grid(perturbed_images[is_adversarial],
                                                            original_targets[is_adversarial],
                                                            n_classes=config.n_classes, n_rows=8)
    display_adv_images = tu.make_grid(display_adv_images, pad_value=2, nrow=10)
    l2_robustness = get_l2_score(perturbed_images, original_images)
    linf_robustness = get_linf_score(perturbed_images, original_images)

    success_rate = float(torch.sum(is_adversarial)) / len(is_adversarial)

    return display_adv_images, display_adv_perturbations, l2_robustness, linf_robustness, success_rate


def get_l2_score(a, b):
    assert a.shape == b.shape
    return torch.median(get_l2_dists(a, b))


def get_linf_score(a, b):
    assert a.shape == b.shape
    return torch.median(torch.max(torch.abs(a - b).flatten(1), dim=1)[0])


def get_l2_dists(a, b):
    return torch.sqrt(torch.sum(((a - b) ** 2).flatten(1), dim=1))
