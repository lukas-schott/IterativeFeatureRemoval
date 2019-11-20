import torch
from iterative_feature_removal import utils as u
from iterative_feature_removal.networks import requires_grad_


def evaluate_net(config, model, data_loader_test):
    model = model.eval()
    with torch.no_grad():
        n_correct, n_total = 0., 0.
        for i, (b, l) in enumerate(data_loader_test):
            u.check_bounds(b)
            b, l = b.to(u.dev()), l.to(u.dev())
            pred = torch.argmax(model(b), dim=1)
            n_correct += torch.sum(pred == l)
            n_total += b.shape[0]
    accuracy = float(n_correct) / n_total
    return accuracy


def mnist_c_evaluation(model, data_loaders, config, writer=None, epoch=None):
    print('mnist_c eval')
    model = model.eval()
    accuracy_all = 0
    accuracy_len_all = 0
    results = {}
    with torch.no_grad():
        for name, data_loader in data_loaders.items():
            n_correct = 0
            for batch_idx, (test_data, test_labels) in enumerate(data_loader):
                test_data, test_labels = test_data.to(u.dev()), test_labels.to(u.dev())
                test_data = test_data.float() / 255.0
                logits = model(test_data)
                pred_label = logits.max(1, keepdim=True)[1]
                n_correct += pred_label.eq(test_labels.view_as(pred_label)).sum().item()

            accuracy_all += n_correct
            accuracy_len_all += len(data_loader.dataset)
            results[name] = float(n_correct) / len(data_loader.dataset)
            if writer is not None:
                writer.add_scalar(f'mnist_c/{name}', float(n_correct) / len(data_loader.dataset), epoch)

        accuracy_all /= accuracy_len_all
        if writer is not None:
            print('Accuracy all: ', accuracy_all)
            writer.add_scalar('test/MNIST-C Accuracy All', accuracy_all, epoch)
    return results


# def get_abs_cosine_similarity(config, model, data_loader_test):
#     model = model.eval()
#
#     with torch.no_grad():
#         n_correct, n_total = 0., 0.
#         for i, (b, l) in enumerate(data_loader_test):
#     grads_wrt_input = torch.autograd.grad([ce_ind], self.model.cached_batches, create_graph=True, retain_graph=True)[0]
#     sensitivity_vectors = grads_wrt_input.view(l.shape[0], self.model.n_groups, -1)
