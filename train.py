import torch
import utils as u
from attacks import calculate_adversarial_perturbations_batch
import time


def train_net(config, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
              data_loader_train, loss_fct=torch.nn.CrossEntropyLoss(),
              adv_training=False, ep_manager=None):
    model = model.train()

    optimizer.zero_grad()

    n_correct, n_total = 0., 0.
    for i, (b, l) in enumerate(data_loader_train):
        b, l = b.to(u.dev()), l.to(u.dev())

        if adv_training:
            b = calculate_adversarial_perturbations_batch(config, model, b, l, train=True).detach()
        if ep_manager is not None:
            b, l = ep_manager.append_with_ep(b, l)
        u.check_bounds(b)
        logits = model(b)
        loss = loss_fct(logits, target=l)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            n_total += b.shape[0]
    accuracy = n_correct / n_total
    return accuracy
