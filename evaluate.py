import torch
import utils as u


def evaluate_net(config, model, data_loader_test):
    model = model.eval()
    with torch.no_grad():
        n_correct, n_total = 0., 0.
        for i, (b, l) in enumerate(data_loader_test):
            u.check_bounds(b)
            assert 0 <= torch.min(b), 'min in batch {torch.min(b)} smaller than 0'
            assert 1 >= torch.max(b), 'max in batch {torch.min(b)} larger than 1'
            b, l = b.to(u.dev()), l.to(u.dev())
            pred = torch.argmax(model(b), dim=1)
            n_correct += torch.sum(pred == l)
            n_total += b.shape[0]

    accuracy = float(n_correct) / n_total
    return accuracy
