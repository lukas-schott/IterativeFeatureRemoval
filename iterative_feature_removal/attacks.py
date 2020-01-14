from torch.utils import data
import torch
from torchvision import utils as tu
from iterative_feature_removal import dataloader as dl, utils as u
from advertorch import attacks as pyatt
import numpy as np
import foolbox
from foolbox.distances import MeanSquaredDistance


def get_attack(model, lp_metric, attack, attack_iter, l2_step_size=0.05, linf_step_size=0.05,
               max_eps_l2=10., max_eps_linf=1., n_classes=10):
    loss_fct = lambda x, y: madry_loss_fct(x, y, margin=0.1)
    if lp_metric == 'l2':
        if attack == 'BIM':
            adversary = pyatt.L2BasicIterativeAttack(model, loss_fn=loss_fct, eps=max_eps_l2, targeted=False,
                                                     eps_iter=l2_step_size,
                                                     nb_iter=attack_iter)
        elif 'PGD' in attack:
            if len(attack) > 3:
                eps_ball = float(attack[4:])
            else:
                eps_ball = max_eps_l2
            adversary = pyatt.L2PGDAttack(model, loss_fn=loss_fct, eps=eps_ball, targeted=False,
                                          eps_iter=l2_step_size,
                                          nb_iter=attack_iter, rand_init=True)
        elif attack == 'CW':
            adversary = pyatt.CarliniWagnerL2Attack(model, n_classes, max_iterations=1000)
        elif attack == 'DDN_L2':
            adversary = pyatt.DDNL2Attack(model, nb_iter=attack_iter)
        else:
            raise Exception(f'attack {attack} not implemented')

    elif lp_metric == 'linf':
        adversary = pyatt.LinfBasicIterativeAttack(model, loss_fn=loss_fct, eps=max_eps_linf, targeted=False,
                                                   eps_iter=linf_step_size,
                                                   nb_iter=attack_iter)
    else:
        raise Exception(f'lp metric {lp_metric} is not defined.')
    return adversary


def get_adversarial_perturbations(config, model, data_loader, adversaries, stop_after_batch=True):
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
        if stop_after_batch:
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
    display_imgs = {}
    # perturbations
    _, display_adv_perturbations, _ = u.get_class_sorted_images(adv_perturbations,
                                                                original_targets,
                                                                n_classes=config.n_classes, n_rows=8)
    display_imgs['perturbations_rescaled'] = tu.make_grid(display_adv_perturbations, pad_value=2, nrow=10)
    # adv images
    display_adv_images, _, _ = u.get_class_sorted_images(perturbed_images,
                                                         original_targets,
                                                         n_classes=config.n_classes, n_rows=8)
    display_imgs['adversarials'] = tu.make_grid(display_adv_images, pad_value=2, nrow=10)
    # originals
    display_original_images, _, _ = u.get_class_sorted_images(original_images,
                                                              original_targets,
                                                              n_classes=config.n_classes, n_rows=8)
    display_imgs['originals'] = tu.make_grid(display_original_images, pad_value=2, nrow=10)
    # gradients
    gradients = get_first_derivative(model, original_images, original_targets)
    _, display_gradient_images, _ = u.get_class_sorted_images(gradients,
                                                              original_targets,
                                                              n_classes=config.n_classes, n_rows=8)
    display_imgs['gradients'] = tu.make_grid(display_gradient_images, pad_value=2, nrow=10)

    gradients_adv_direction = get_first_derivative(model, original_images, original_targets, adv_direction=True)
    _, display_adv_gradient_images, _ = u.get_class_sorted_images(gradients_adv_direction,
                                                                  original_targets,
                                                                  n_classes=config.n_classes, n_rows=8)
    display_imgs['gradients_adv_direction'] = tu.make_grid(display_adv_gradient_images, pad_value=2, nrow=10)

    l2_robustness, l2_accuracy = get_l2_scores(perturbed_images, original_images, is_adversarial,
                                               eps_threshold=config.epsilon_threshold_accuracy_l2)
    linf_robustness, linf_accuracy = get_linf_scores(perturbed_images, original_images, is_adversarial,
                                                     eps_threshold=config.epsilon_threshold_accuracy_linf)
    success_rate = float(torch.sum(is_adversarial)) / len(is_adversarial)

    perturbed_data = [perturbed_images.detach().cpu(), original_targets]
    return display_imgs, l2_robustness, l2_accuracy, linf_robustness, linf_accuracy, success_rate, perturbed_data


def generate_new_dataset(config, model, data_loader, adversary):
    adv_perturbations, perturbed_images, original_images, original_targets, is_adversarial = \
        get_adversarial_perturbations(config, model, data_loader, adversaries=[adversary],
                                      stop_after_batch=False)
    print('len ', len(is_adversarial), 'sucess rate', float(is_adversarial.shape[0]) / torch.sum(is_adversarial))
    data_loader = dl.create_new_dataset(config, perturbed_images, original_images, original_targets,
                                        is_adversarial, data_loader)
    return data_loader


def get_l2_scores(a, b, is_adversarial=None, eps_threshold=1.5):
    assert a.shape == b.shape
    l2_dists = get_l2_dists(a, b)
    if is_adversarial is not None:
        l2_dists[is_adversarial.bitwise_not()] = 100
        robust_accuracy = float(torch.sum(l2_dists > eps_threshold)) / l2_dists.shape[0]
    return torch.median(l2_dists), robust_accuracy


def get_linf_scores(a, b, is_adversarial=None, eps_threshold=0.3):
    assert a.shape == b.shape
    linf_dists = torch.max(torch.abs(a - b).flatten(1), dim=1)[0]
    if is_adversarial is not None:
        linf_dists[is_adversarial.bitwise_not()] = 1
        robust_accuracy = float(torch.sum(linf_dists > eps_threshold)) / linf_dists.shape[0]
    return torch.median(linf_dists), robust_accuracy


def get_l2_dists(a, b):
    return torch.sqrt(torch.sum(((a - b) ** 2).flatten(1), dim=1))


def madry_loss_fct(logits, l, margin=50.):
    true_logit = logits[range(len(l)), l]
    mask = (1 - u.label_2_onehot(l)).type(torch.bool)
    false_logit = torch.max(logits[mask].view(l.shape[0], 9), dim=1)[0]
    loss = - torch.sum(torch.relu(true_logit - false_logit + margin))
    return loss


def orthogonal_projection(original_imgs, perturbed_imgs):
    n_feats = np.prod(original_imgs.shape[1:])
    # make direction linear and unit length
    perturbations = original_imgs - perturbed_imgs
    perturbations /= torch.norm(perturbations.view((-1, n_feats)),  dim=1)[:, None, None, None] + 0.0000001
    lambdas = torch.sum(perturbations.view((-1, n_feats)) * original_imgs.view(-1, n_feats), dim=1)
    new_imgs = torch.clamp(original_imgs - lambdas[:, None, None, None] * perturbations, 0, 1)
    return new_imgs


def get_first_derivative(model, b, l, adv_direction=False):
    l = l.to(u.dev())
    model.eval()
    b.requires_grad_(True)
    logits = model(b.to(u.dev()))
    if adv_direction:
        best_other_logit, _ = u.get_best_non_target_logit(logits, l)
        b_inds = range(len(l))
        loss = -(logits[b_inds, l] - best_other_logit).sum()
    else:
        loss = -logits[range(len(l)), l].sum()
    gradients = torch.autograd.grad([loss], b, retain_graph=False, create_graph=False)[0]
    return gradients


def test_under_shift_rotation(config, model, data_loader):
    print('shift erot attack')
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=config.attack_batch_size, shuffle=False)

    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    shift, rot = config.attack_shift, config.attack_rot

    kwargs = {'x_shift_limits': (-shift, shift), 'y_shift_limits': (-shift, shift),  'angular_limits': (-rot, rot)}

    attack = foolbox.attacks.SpatialAttack(fmodel)

    for b, l in sorted_data_loader:
        b, l = b.to(u.dev()), l.to(u.dev())
        break
    adversarials = torch.tensor(attack(b.cpu().numpy(), l.cpu().numpy(), **kwargs)).to(u.dev())

    mask = (adversarials != adversarials).type(torch.bool)
    adversarials[mask] = b[mask]
    pred = torch.argmax(model(adversarials), dim=1)
    is_adversarial = (pred != l).float()
    print('shift rot attack', is_adversarial.shape,  is_adversarial.mean())
    return 1. - is_adversarial.mean()


def run_boundary_attack(model, data_loader, n_iter):
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=100, shuffle=False)  # slow --> only small bs

    model = model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=10)
    attack = foolbox.attacks.BoundaryAttack(fmodel, distance=MeanSquaredDistance)
    # attack = foolbox.attacks.L2BasicIterativeAttack(fmodel, distance=MeanSquaredDistance)  # to debug

    for b, l in sorted_data_loader:
        b, l = b.to(u.dev()), l.to(u.dev())
        break

    # requires at least foolbox version 2.3.0
    adversarials = torch.tensor(attack(b.cpu().numpy(), l.cpu().numpy(), iterations=n_iter)).to(u.dev())

    mask = (adversarials != adversarials).type(torch.bool)  # detect nans
    adversarials[mask] = b[mask]
    is_adversarial = (torch.argmax(model(adversarials), dim=1) != l).float()

    dists = torch.sqrt(torch.sum((adversarials.flatten(1) - b.flatten(1))**2, dim=1))
    dists[is_adversarial.type(torch.bool).bitwise_not()] = 1000

    print(int(torch.sum(is_adversarial)), 'out of', len(is_adversarial), 'sucessfully attacked')
    print('distance', torch.median(dists))   # vanilla net has 1.1
    adversarials[mask] = 0
    adversarials = tu.make_grid(adversarials, pad_value=2, nrow=10)
    return torch.median(dists), adversarials
