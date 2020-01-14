import torch
from iterative_feature_removal import utils as u
from iterative_feature_removal import train as t
from torch.utils import data
from matplotlib import pyplot as plt


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


def get_similarity_measures(config, model, data_loader):
    model = model.eval()
    n_redundant, n_classes = model.n_redundant, config.n_classes
    sorted_data_loader = data.DataLoader(data_loader.dataset, batch_size=config.attack_batch_size, shuffle=False)

    for i, (b, l) in enumerate(sorted_data_loader):
        b, l = b.to(u.dev()), l.to(u.dev())
        break

    bs = b.shape[0]
    l_ind = l[:, None].expand((bs, n_redundant)).flatten()    # same as n_redundant

    _, ind_logits = model(b, return_individuals=True)
    assert model.cached_batches.shape[:2] == (bs, n_redundant)
    selected_logits = t.select_logits(ind_logits.view(bs * n_redundant, n_classes), l_ind,
                                      'target', n_classes).view(bs, n_redundant)
    sensitivity_vectors = t.get_grads_wrt_input(
        model, selected_logits, False, create_graph=False).view(bs, n_redundant, -1).detach().abs()
    abs_cosine_similarity = t.calc_similarity_estimator(sensitivity_vectors).mean(dim=0)
    abs_scalar_product_similarity = t.calc_similarity_estimator(sensitivity_vectors,
                                                                similarity_measure='scalar_prod_abs').mean(dim=0)
    return abs_cosine_similarity.cpu(), abs_scalar_product_similarity.cpu(), sensitivity_vectors.abs().mean()


def plot_similarities(config, model, data_loader, writer, epoch):
    abs_cosine_similarity, abs_scalar_product_similarity, grads = get_similarity_measures(config, model, data_loader)

    writer.add_scalar('test/gradient_magnitude', grads, epoch)
    n_redundant = model.n_redundant
    mask = torch.triu(torch.ones(n_redundant, n_redundant), diagonal=1).type(torch.bool)
    for name, similarity in zip(['abs_scalar_product_similarity', 'abs_cosine_similarity'],
                                [abs_scalar_product_similarity, abs_cosine_similarity]):
        mean_sim = similarity[mask].mean()
        writer.add_scalar(f'test/{name}', mean_sim, epoch)
        fig = u.plot_similarity_matrix(similarity)

        writer.add_figure(f'test_sim_matrix/{name}', fig, epoch)


