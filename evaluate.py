import torch
import utils as u
from attacks import calculate_adversarial_perturbations_batch
from torch.utils import data


def evaluate_net(config, model, data_loader, adv_training):
    model = model.eval()
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=10000, shuffle=False)

    with torch.no_grad():
        n_correct, n_total = 0., 0.
        for i, (b, l) in enumerate(sorted_data_loader):
            b, l = b.to(u.dev()), l.to(u.dev())
            if adv_training:
                b = calculate_adversarial_perturbations_batch(config, model, b, l, rand_init=False).detach()

            u.check_bounds(b)
            assert 0 <= torch.min(b), 'min in batch {torch.min(b)} smaller than 0'
            assert 1 >= torch.max(b), 'max in batch {torch.min(b)} larger than 1'
            pred = torch.argmax(model(b), dim=1)
            n_correct += torch.sum(pred == l)
            n_total += b.shape[0]
            break

    accuracy = float(n_correct) / n_total
    return accuracy
