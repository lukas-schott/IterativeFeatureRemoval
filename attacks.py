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
        break
    accuracy = float(n_correct) / n_total

    display_noise_images, _, _ = u.get_indices_for_class_grid(b_plot, l_plot, n_classes=config.n_classes, n_rows=5)
    display_noise_images = tu.make_grid(display_noise_images, pad_value=2, nrow=5)

    return accuracy, display_noise_images


def calculate_adversarial_perturbations_batch(config, model, b, l, train):
    adversary = get_attack(model, lp_metric=config.lp_metric, eps=config.adv_epsilon, config=config, train=train)
    adv_untargeted = adversary.perturb(b, l)
    return adv_untargeted


def get_attack(model, lp_metric, eps, config, train=True):
    loss_fct = lambda x, y: madry_loss_fct(x, y, margin=(50 if train else 0.1))

    if lp_metric == 'l2':
        adversary = pyatt.L2PGDAttack(model, loss_fn=loss_fct, eps=eps, targeted=False, rand_init=train,
                                      eps_iter=0.05, nb_iter=(config.attack_iter if train else 100))
    elif lp_metric == 'linf':
        adversary = pyatt.LinfPGDAttack(model, loss_fn=loss_fct, eps=eps, targeted=False, rand_init=train,
                                        eps_iter=config.pgd_step_size, nb_iter=(config.attack_iter if train else 100))
    else:
        raise Exception(f'lp metric {lp_metric} is not defined.')
    return adversary


def get_adversarial_perturbations(config, model, data_loader, lp_metric, eps, train=True):
    model.eval()
    adversary = get_attack(model, lp_metric, eps, config, train)

    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=1000, shuffle=False)
    perturbed_images = []
    is_adversarial = []
    original_images = []
    original_targets = []
    for i, (b, l) in enumerate(sorted_data_loader):
        b, l = b.to(u.dev()), l.to(u.dev())
        u.check_bounds(b)
        adversarials = adversary.perturb(b, l).detach()
        with torch.no_grad():
            is_adversarial += [(torch.argmax(model(adversarials), dim=1) != l).cpu().type(torch.bool)]
        perturbed_images += [adversarials.cpu()]

        original_images += [b.cpu()]
        original_targets += [l.cpu()]
        break
    perturbed_images = torch.cat(perturbed_images, dim=0)
    is_adversarial = torch.cat(is_adversarial, dim=0)  # ignore it if attack failed
    original_images = torch.cat(original_images, dim=0)
    original_targets = torch.cat(original_targets, dim=0)
    return perturbed_images, original_images, original_targets, is_adversarial


def evaluate_robustness(config, model, data_loader, lp_metric, eps, train):
    perturbed_images, original_images, original_targets, is_adversarial = \
        get_adversarial_perturbations(config, model, data_loader, lp_metric, eps=eps, train=train)
    adv_perturbations = perturbed_images - original_images
    _, display_adv_perturbations, _ = u.get_indices_for_class_grid(adv_perturbations[is_adversarial],
                                                                   original_targets[is_adversarial],
                                                                   n_classes=config.n_classes, n_rows=8)
    display_adv_perturbations = tu.make_grid(display_adv_perturbations, pad_value=2, nrow=10)

    display_adv_images, _, _ = u.get_indices_for_class_grid(perturbed_images[is_adversarial],
                                                            original_targets[is_adversarial],
                                                            n_classes=config.n_classes, n_rows=8)
    display_adv_images = tu.make_grid(display_adv_images, pad_value=2, nrow=10)
    l2_robustness, l2_accuracy = get_l2_score(perturbed_images, original_images, is_adversarial)
    linf_robustness, linf_accuracy = get_linf_score(perturbed_images, original_images, is_adversarial)

    success_rate = float(torch.sum(is_adversarial)) / len(is_adversarial)
    return display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, linf_accuracy, \
           success_rate


def get_l2_score(a, b, is_adversarial=None):
    assert a.shape == b.shape
    l2_dists = get_l2_dists(a, b)
    if is_adversarial is not None:
        l2_dists[is_adversarial.bitwise_not()] = 10
        robust_accuracy = float(torch.sum(l2_dists > 1.5)) / l2_dists.shape[0]
    return torch.median(l2_dists), robust_accuracy


def get_linf_score(a, b, is_adversarial=None):
    assert a.shape == b.shape
    linf_dists = torch.max(torch.abs(a - b).flatten(1), dim=1)[0]
    if is_adversarial is not None:
        linf_dists[is_adversarial.bitwise_not()] = 1
        robust_accuracy = float(torch.sum(linf_dists > 0.3)) / linf_dists.shape[0]
    return torch.median(linf_dists), robust_accuracy


def get_l2_dists(a, b):
    return torch.sqrt(torch.sum(((a - b) ** 2).flatten(1), dim=1))


def madry_loss_fct(logits, l, margin=50):
    true_logit = logits[range(len(l)), l]
    mask = (1 - u.label_2_onehot(l)).type(torch.bool)
    false_logit = torch.max(logits[mask].view(l.shape[0], 9), dim=1)[0]
    loss = - torch.sum(torch.relu(true_logit - false_logit + margin))
    return loss
