import torch
from torch.nn import functional as F
import utils as u
from attacks import calculate_adversarial_perturbations_batch


def train_net(config, model, optimizer, data_loader_train, loss_fct=torch.nn.CrossEntropyLoss(),
              adv_training=False):
    model = model.train()

    optimizer.zero_grad()

    n_correct, n_total = 0., 0.
    for i, (b, l) in enumerate(data_loader_train):
        b, l = b.to(u.dev()), l.to(u.dev())

        if adv_training:
            b = calculate_adversarial_perturbations_batch(config, model, b, l, train=True).detach()

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
