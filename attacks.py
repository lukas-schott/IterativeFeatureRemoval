import foolbox
from foolbox import models, batch_attacks, criteria
from foolbox import run_parallel, run_sequential
import numpy as np
from torch.utils import data
import utils as u
import torch
from torchvision import utils as tu
import dataloader as dl

print('foolbox', foolbox.__version__, foolbox.__path__)


def calculate_adversarial_perturbations(config, model, data_loader):
    print('running adv attack')
    print()
    model.eval()
    fmodel = models.PyTorchModel(model,  # returns logits in shape (bs, n_classes)
                                 bounds=(0., 1.), num_classes=config.n_classes,
                                 device=u.dev())
    criterion = criteria.Misclassification()
    attack_fn = batch_attacks.L2BasicIterativeAttack
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=10000, shuffle=False)
    adversarial_images = []
    is_adversarial = []
    original_images = []
    original_targets = []
    for i, (b, l) in enumerate(sorted_data_loader):
        u.check_bounds(b)
        print(f'i {i} out of ', len(sorted_data_loader))

        adversarials = run_parallel(attack_fn, fmodel, criterion, b.numpy(), l.numpy(),
                                    distance=foolbox.distances.MeanSquaredDistance)
        adversarial_images += [a.perturbed if a.perturbed is not None else a.unperturbed for a in adversarials]
        is_adversarial.append(np.array([a.adversarial_class is not None for a in adversarials]))

        original_images += [b]
        original_targets += [l]

    perturbed_images = np.concatenate(adversarial_images, axis=0)[:, None, :, :]
    is_adversarial = np.concatenate(is_adversarial, axis=0)   # ignore it if attack failed
    original_images = np.concatenate(original_images, axis=0)
    original_targets = np.concatenate(original_targets, axis=0)
    return perturbed_images, original_images, original_targets, is_adversarial


def create_adversarial_dataset(config, model, data_loader, keep_data_loader=True):
    perturbed_images, original_images, original_targets, is_adversarial = \
        calculate_adversarial_perturbations(config, model, data_loader)
    adv_perturbations = torch.from_numpy(perturbed_images - original_images)
    _, display_adv_perturbations = u.get_indices_for_class_grid(adv_perturbations[is_adversarial],
                                                                data_loader.dataset.targets[is_adversarial],
                                                                n_classes=config.n_classes, n_rows=8)
    display_adv_perturbations = tu.make_grid(display_adv_perturbations, pad_value=2, nrow=10)

    display_adv_images, _ = u.get_indices_for_class_grid(torch.from_numpy(perturbed_images[is_adversarial]),
                                                         data_loader.dataset.targets[is_adversarial],
                                                         n_classes=config.n_classes, n_rows=8)
    display_adv_images = tu.make_grid(display_adv_images, pad_value=2, nrow=10)
    l2_robustness = get_l2_score(perturbed_images, original_images)

    if not keep_data_loader:
        dl.create_new_dataset(config, perturbed_images, original_images, original_targets, is_adversarial,
                              data_loader)

    success_rate = float(np.sum(is_adversarial)) / len(is_adversarial)

    return display_adv_images, display_adv_perturbations, l2_robustness, success_rate


def get_l2_score(a, b):
    n_feats = np.prod(b.shape[1:])
    assert np.all(a.shape == b.shape)
    return np.mean(np.sum(((a - b) ** 2).reshape(-1, n_feats), axis=1))





