import torch
from iterative_feature_removal import utils as u


def train_net(config, model, optimizer, data_loader_train, loss_fct=torch.nn.CrossEntropyLoss()):
    model = model.train()

    optimizer.zero_grad()

    n_correct, n_total = 0., 0.
    for i, (b, l) in enumerate(data_loader_train):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        logits = model(b)
        loss = loss_fct(logits, target=l)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            n_total += b.shape[0]
    accuracy = n_correct / n_total
    return accuracy
