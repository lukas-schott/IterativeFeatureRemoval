import torch
from torch.nn import functional as F
from iterative_feature_removal.train import get_trainer, get_optimizer


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
               replace_best: bool = False):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer, 'epoch': epoch},
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
