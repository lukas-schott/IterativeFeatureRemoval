import torch
from torch.nn import functional as F
from iterative_feature_removal.train import get_trainer, get_optimizer
from matplotlib import pyplot as plt
from torch.utils import data


def dev():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return


def check_bounds(b, minimum=0, maximum=1):
    assert minimum <= torch.min(b), f'min val in batch is: {torch.min(b)}'
    assert maximum >= torch.max(b), f'max val in batch is: {torch.max(b)}'
    # assert 0.5 <= torch.max(b)
    # assert 0.5 >= torch.min(b)


def get_class_sorted_images(data, labels, n_classes=10, n_rows=8):
    images = []
    for i in range(n_classes):
        images_per_class = data[labels == i][:n_rows]
        n_successful = images_per_class.shape[0]
        images_per_class_new = torch.zeros((n_rows, *data.shape[1:]), dtype=images_per_class.dtype)     # fill with 0s
        images_per_class_new[:n_successful] = images_per_class
        images += [images_per_class_new]
    images = torch.stack(images, dim=1).reshape((n_rows * n_classes, *data.shape[1:]))
    images_rescaled = rescale_image_to_01(images)
    labels = torch.arange(n_classes).repeat(n_rows).T.flatten()
    return images, images_rescaled, labels


def get_loss_fct(config):
    if config.loss_fct == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif config.loss_fct == 'soft_ce':
        return soft_ce
    else:
        raise Exception(f'loss function {config.loss_fct} is not defined')


def soft_ce(logits, target, eps=0.1):
    n_class = logits.size(1)

    one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(logits, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return torch.mean(loss)


def rescale_image_to_01(images: torch.Tensor):
    assert len(images.shape) == 4
    images_rescaled = images - torch.min(images.flatten(1), dim=1)[0][:, None, None, None]
    images_rescaled /= torch.max(images_rescaled.flatten(1), dim=1)[0][:, None, None, None]
    return images_rescaled


def label_2_onehot(l: torch.Tensor, n_classes: int = 10):
    y_onehot = torch.empty((len(l), n_classes), device=dev()).zero_()
    return y_onehot.scatter_(1, l[:, None], 1)


def save_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, save_path: str, epoch: int,
               replace_best: bool = False, config=None):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer, 'epoch': epoch, 'config': config},
               save_path + f'/save_model_{epoch}.pt')
    if replace_best:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer, 'epoch': epoch},
                   save_path + '/save_model_best.pt')


def update_for_greedy_training(trainer, model, optimizer, config, epoch, data_loaders, loss_fct):
    if epoch == 0:
        config.training_mode = 'normal'
        optimizer = get_optimizer(config, model.networks[0].parameters())
        model.n_redundant = 1
        Trainer = get_trainer(config)
        trainer = Trainer(model, data_loaders['train'], optimizer, config, loss_fct)

        config.training_mode = 'redundancy'
    if epoch > 1:
        if model.n_redundant == config.n_redundant:
            exit()
        optimizer = get_optimizer(config, model.networks[model.n_redundant].parameters())
        model.n_redundant += 1
        Trainer = get_trainer(config)
        trainer = Trainer(model, data_loaders['train'], optimizer, config, loss_fct)

    return trainer, model, optimizer, config


def get_best_non_target_logit(logits, l):
    top_2 = torch.topk(logits, 2).indices
    is_true_logit_best_mask = top_2[:, 0] == l
    best_other_indices = top_2[:, 0]
    best_other_indices[is_true_logit_best_mask] = top_2[:, 1][is_true_logit_best_mask]
    best_other_logits = logits[range(len(l)), best_other_indices]
    return best_other_logits, best_other_indices


def plot_similarity_matrix(similarity, vmin=0, vmax=1, names=None, xlabel='', ylabel=''):
    n = similarity.shape[0]
    fig, ax = plt.subplots(figsize=(n, n))

    ax.matshow(similarity, cmap='Greens', vmin=vmin, vmax=vmax)
    for i in range(n):
        for j in range(n):
            c = similarity[j, i]
            ax.text(i, j, f'{c:0.5f}', va='center', ha='center')
    if names is not None:
        ax.set_xticklabels([''] + names)
        ax.set_yticklabels([''] + names)
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.axis('off')
    return fig


def get_fixed_batch(data_loader, bs=1000):
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=bs, shuffle=False)  # slow --> only small bs

    for b, l in sorted_data_loader:
        b, l = b.to(dev()), l.to(dev())
        break
    return b, l


def fix_perturbation_size(x0, delta, epsilon):
    """
    returns eta such that
        norm(clip(x0 + eta * delta, 0, 1)) == epsilon
    assumes x0 and delta to have a batch dimension
    and epsilon to be a scalar
    """
    N, ch, nx, ny = x0.shape
    assert delta.shape[0] == N

    delta2 = delta.pow(2).flatten(1)
    space = torch.where(delta >= 0, 1 - x0, x0).flatten(1)
    f2 = space.pow(2) / torch.max(delta2, 1e-20 * torch.ones_like(delta2))
    f2_sorted, ks = torch.sort(f2, dim=-1)
    m = torch.cumsum(delta2.gather(dim=-1, index=ks.flip(dims=(1,))), dim=-1).flip(dims=(1,))
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = torch.cat((f2_sorted[:, :1], dx), dim=-1)
    dy = m * dx
    y = torch.cumsum(dy, dim=-1)
    c = y >= epsilon ** 2

    # work-around to get first nonzero element in each row
    f = torch.arange(c.shape[-1], 0, -1, device=c.device)
    v, j = torch.max(c.long() * f, dim=-1)

    rows = torch.arange(0, N)

    eta2 = f2_sorted[rows, j] - (y[rows, j] - epsilon ** 2) / m[rows, j]
    # it can happen that for certain rows even the largest j is not large enough
    # (i.e. v == 0), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem)
    eta2 = torch.where(v == 0, f2_sorted[:, -1], eta2)
    eta = torch.sqrt(eta2)
    eta = eta.reshape((-1,) + (1,) * (len(x0.shape) - 1))

    # xp = torch.clamp(x0 + eta * delta, 0, 1)
    # l2 = torch.norm((xp - x0).reshape((N, -1)), dim=-1)

    return torch.clamp(eta * delta + x0, 0, 1).view(N, ch, nx, ny)