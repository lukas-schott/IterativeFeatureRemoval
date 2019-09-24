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


class LinfPGDAttack:
    def __init__(self, model, epsilon=0.3, n_iter=40, step_size=0.01, random_start=True, loss_func='xent'):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.n_ter = n_iter
        self.step_size = step_size
        self.rand = random_start

        # if loss_func == 'xent':
        #     loss = model.xent
        # elif loss_func == 'cw':
        #     label_mask = tf.one_hot(model.y_input,
        #                             10,
        #                             on_value=1.0,
        #                             off_value=0.0,
        #                             dtype=tf.float32)
        #     correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
        #     wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
        #                                 - 1e4*label_mask, axis=1)
        #     loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        # else:
        #     print('Unknown loss function. Defaulting to cross-entropy')
        #     loss = model.xent
        #
        # self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, b, l):
        self.model.eval()
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""

        if self.rand:
            x = b + torch.empty(b.shape, device=u.dev()).uniform_(-self.epsilon, self.epsilon)
            inp_x = torch.clamp(x, 0, 1).clone()    # ensure valid pixel range
        else:
            inp_x = b.clone()

        optim = torch.optim.SGD([inp_x], lr=self.step_size, momentum=0)

        inp_x.requires_grad = True
        print('n iter', self.n_ter, 'step', self.step_size)
        for i in range(self.n_ter):
            logits = self.model(inp_x)
            true_logit = logits[range(len(l)), l]
            mask = (1 - u.label_2_onehot(l)).type(torch.bool)
            false_logit = torch.sum(logits[mask].view(inp_x.shape[0], -1), dim=1)
            loss = torch.sum(torch.relu(true_logit - false_logit + 50))

            if inp_x.grad is not None:
                inp_x.grad.zero_()
            loss.backward()

            inp_x.data += self.step_size * torch.sign(inp_x.grad)

            # inp_x.data = torch.max(torch.min(inp_x, b + self.epsilon), b - self.epsilon)

            inp_x.data = torch.clamp(inp_x, 0, 1)       # ensure valid pixel range

        return inp_x.detach()
