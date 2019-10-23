from torch.utils import data
import torch
from torchvision import utils as tu
from iterative_feature_removal import dataloader as dl, utils as u
from advertorch import attacks as pyatt


def get_attack(model, lp_metric, config, attack, max_eps_l2=10., max_eps_linf=1. ):
    loss_fct = lambda x, y: madry_loss_fct(x, y, margin=0.1)
    if lp_metric == 'l2':
        if attack == 'BIM':
            adversary = pyatt.L2BasicIterativeAttack(model, loss_fn=loss_fct, eps=max_eps_l2, targeted=False,
                                                     eps_iter=config.attack_l2_step_size,
                                                     nb_iter=config.attack_iter)
        elif 'PGD' in attack:
            eps_ball = float(attack[4:])
            adversary = pyatt.L2PGDAttack(model, loss_fn=loss_fct, eps=eps_ball, targeted=False,
                                          eps_iter=config.attack_l2_step_size,
                                          nb_iter=config.attack_iter, rand_init=True)
        elif attack == 'CW':
            adversary = pyatt.CarliniWagnerL2Attack(model, config.n_classes, max_iterations=50)
        elif attack == 'DNN_L2':
            adversary = pyatt.DDNL2Attack(model, nb_iter=config.attack_iter)
        elif attack == 'L-BFGS':
            adversary = pyatt.LBFGSAttack(model, config.n_classes, batch_size=config.attack_batch_size,
                                          loss_fn=loss_fct)
        else:
            raise Exception(f'attack {attack} not implemented')

    elif lp_metric == 'linf':
        adversary = pyatt.LinfBasicIterativeAttack(model, loss_fn=loss_fct, eps=max_eps_linf, targeted=False,
                                                   eps_iter=config.attack_linf_step_size,
                                                   nb_iter=config.attack_iter)
    else:
        raise Exception(f'lp metric {lp_metric} is not defined.')
    return adversary


def get_adversarial_perturbations(config, model, data_loader, adversaries):
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=config.attack_batch_size, shuffle=False)
    perturbed_images = []
    is_adversarial = []
    original_images = []
    original_targets = []

    for i, (b, l) in enumerate(sorted_data_loader):
        b, l = b.to(u.dev()), l.to(u.dev())
        u.check_bounds(b)

        adversarials, is_adversarial_batch = run_attacks_single_batch(adversaries, model, b, l, get_l2_dists)

        with torch.no_grad():
            is_adversarial += [is_adversarial_batch.cpu()]
            perturbed_images += [adversarials.cpu()]
            original_images += [b.cpu()]
            original_targets += [l.cpu()]
        break
    perturbed_images = torch.cat(perturbed_images, dim=0)
    is_adversarial = torch.cat(is_adversarial, dim=0)   # ignore it if attack failed
    original_images = torch.cat(original_images, dim=0)
    original_targets = torch.cat(original_targets, dim=0)
    adv_perturbations = perturbed_images - original_images
    return adv_perturbations, perturbed_images, original_images, original_targets, is_adversarial


def run_attacks_single_batch(adversaries, model, b, l, norm_fct):
    max_val = b[0].numel()
    inds_b = list(range(b.shape[0]))
    all_l2s = []
    for i, adversary in enumerate(adversaries):
        adversarials = adversary.perturb(b, l).detach()
        is_adversarial = (torch.argmax(model(adversarials), dim=1) != l).cpu().type(torch.bool)
        lp_dist = norm_fct(b, adversarials)
        lp_dist[is_adversarial.bitwise_not()] = max_val   # max l2 val'
        all_l2s.append(lp_dist)

        if i == 0:
            lp_bests = lp_dist
            is_adversarial_best = is_adversarial
            adversarial_best = adversarials
        else:
            inds = torch.argmin(torch.stack([lp_bests, lp_dist], dim=1), dim=1)
            is_adversarial_best = torch.stack([is_adversarial_best, is_adversarial], dim=1)[inds_b, inds]
            adversarial_best = torch.stack([adversarial_best, adversarials], dim=1)[inds_b, inds]
    return adversarial_best, is_adversarial_best


def evaluate_robustness(config, model, data_loader, adversaries):
    model.eval()

    adv_perturbations, perturbed_images, original_images, original_targets, is_adversarial = \
        get_adversarial_perturbations(config, model, data_loader, adversaries=adversaries)

    _, display_adv_perturbations, _ = u.get_indices_for_class_grid(adv_perturbations,
                                                                   original_targets,
                                                                   n_classes=config.n_classes, n_rows=8)
    display_adv_perturbations = tu.make_grid(display_adv_perturbations, pad_value=2, nrow=10)

    display_adv_images, _, _ = u.get_indices_for_class_grid(perturbed_images,
                                                            original_targets,
                                                            n_classes=config.n_classes, n_rows=8)
    display_adv_images = tu.make_grid(display_adv_images, pad_value=2, nrow=10)
    l2_robustness, l2_accuracy = get_l2_scores(perturbed_images, original_images, is_adversarial)
    linf_robustness, linf_accuracy = get_linf_scores(perturbed_images, original_images, is_adversarial)

    success_rate = float(torch.sum(is_adversarial)) / len(is_adversarial)

    return display_adv_images, display_adv_perturbations, l2_robustness, l2_accuracy, linf_robustness, linf_accuracy, \
           success_rate, data_loader


def get_l2_scores(a, b, is_adversarial=None):
    assert a.shape == b.shape
    l2_dists = get_l2_dists(a, b)
    if is_adversarial is not None:
        l2_dists[is_adversarial.bitwise_not()] = 10
        robust_accuracy = float(torch.sum(l2_dists > 1.5)) / l2_dists.shape[0]
    return torch.median(l2_dists), robust_accuracy


def get_linf_scores(a, b, is_adversarial=None):
    assert a.shape == b.shape
    linf_dists = torch.max(torch.abs(a - b).flatten(1), dim=1)[0]
    if is_adversarial is not None:
        linf_dists[is_adversarial.bitwise_not()] = 1
        robust_accuracy = float(torch.sum(linf_dists > 0.3)) / linf_dists.shape[0]
    return torch.median(linf_dists), robust_accuracy


def get_l2_dists(a, b):
    return torch.sqrt(torch.sum(((a - b) ** 2).flatten(1), dim=1))


def madry_loss_fct(logits, l, margin=50.):
    true_logit = logits[range(len(l)), l]
    mask = (1 - u.label_2_onehot(l)).type(torch.bool)
    false_logit = torch.max(logits[mask].view(l.shape[0], 9), dim=1)[0]
    loss = - torch.sum(torch.relu(true_logit - false_logit + margin))
    return loss
