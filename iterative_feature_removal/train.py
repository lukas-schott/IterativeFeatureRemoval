import torch
from torch.nn import functional as F
from iterative_feature_removal import utils as u
from iterative_feature_removal.attacks import get_attack, orthogonal_projection
from torch import optim
from iterative_feature_removal.networks import requires_grad_
import matplotlib.pyplot as plt


def get_trainer(config):
    if config.training_mode == 'normal':
        print('Vanilla trainer')
        return Trainer
    elif config.training_mode == 'adversarial' and not config.activation_matching:
        print('Adversarial trainer')
        return AdversarialTrainer
    elif config.training_mode == 'adversarial' and config.activation_matching:
        print('Adversarial activation matching trainer')
        return ActivationMatchingAdversarialTrainer
    elif config.training_mode == 'adversarial_projection' and not config.activation_matching:
        print('Adversarial projection trainer')
        return AdversarialOrthogonalProjectionTrainer
    elif config.training_mode == 'adversarial_projection' and config.activation_matching:
        print('Adversarial projection activation matching trainer')
        return ActivationMatchingAdversarialOrthogonalProjectionTrainer
    elif config.training_mode == 'redundancy':
        print('Redundancy Trainer')
        return RedundancyTrainer
    else:
        print('mode', config.training_mode)
        raise NotImplementedError


def get_optimizer(config, parameter_gen):
    if config.optimizer == 'adam':
        return optim.Adam(parameter_gen, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        return optim.SGD(parameter_gen, lr=config.lr, momentum=config.momentum,
                         weight_decay=config.weight_decay)
    else:
        print('optmizer', config.optimizer)
        raise NotImplementedError


class Trainer:
    def __init__(self, model: torch.nn.Module,
                 data_loader,
                 optimizer: optim.Adam,
                 config,
                 class_loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        self.class_loss_fct = class_loss_fct
        self.n_correct = 0
        self.optimizer = optimizer
        self.model = model      # pointer to model
        assert id(model) == id(self.model)  # assure its pointer
        self.data_loader = data_loader
        self.config = config
        self.n_correct = 0
        self.n_total = 0
        self.epoch_stats = {}

    def reset_counters(self):
        self.n_correct = 0.
        self.n_total = 0.
        self.epoch_stats = {'cross_entropy': []}

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        self.reset_counters()

        for i, (b, l) in enumerate(self.data_loader):
            self.train_iter(b, l, epoch=epoch)

        accuracy = self.n_correct / self.n_total
        return accuracy

    def forward(self, b):
        return self.model(b)

    def loss(self, logits, l):
        assert not torch.isnan(logits).any().bool().item()
        ce_loss = self.class_loss_fct(logits, target=l)
        self.epoch_stats['cross_entropy'].append(ce_loss)
        return ce_loss

    def preprocess_data(self, b, l, epoch=None):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        return b, l

    def train_iter(self, b, l, epoch=None):
        b, l = self.preprocess_data(b, l, epoch=epoch)
        output = self.forward(b)
        loss = self.loss(output, l)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.track_statistics(b, l, output)

    def track_statistics(self, b, l, logits):
        with torch.no_grad():
            self.n_correct += (torch.argmax(logits, dim=1) == l).float().sum()
            self.n_total += b.shape[0]

    def write_stats(self, writer, epoch):
        writer.add_scalar('train/accuracy', self.n_correct / self.n_total, epoch)
        writer.add_scalar('train/cross_entropy_loss', torch.mean(torch.stack(self.epoch_stats['cross_entropy'])), epoch)


class ActivationMatchingTrainer(Trainer):
    def forward(self, b):
        return self.model(b, return_activations=True)

    def preprocess_data(self, b, l, epoch=None):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        b = b.view(b.shape[0] * b.shape[1], *b.shape[2:])
        l = torch.stack([l, l], dim=1).flatten()
        return b, l

    def loss(self, output, l):
        logits, activations = output
        assert not torch.isnan(logits).any().bool().item()
        class_loss = self.class_loss_fct(logits, l)
        activation_loss = 0
        for activation in activations:
            activation_loss += torch.mean((activation[::2] - activation[1::2])**2)
        loss = class_loss + self.config.siamese_activations_weight * activation_loss
        return loss

    def track_statistics(self, b, l, outputs):
        logits, _ = outputs
        super().track_statistics(b, l, logits)


class AdversarialTrainer(Trainer):
    def preprocess_data(self, b, l, epoch=None):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        logits = self.model(b)
        assert not torch.isnan(logits).any().bool().item()

        if epoch < self.config.epoch_start_adv_train:
            return b, l

        requires_grad_(self.model, False)
        adversary = get_attack(self.model, self.config.lp_metric, self.config.attack_train_name,
                               self.config.attack_train_iter,
                               l2_step_size=self.config.attack_train_l2_step_size,
                               max_eps_l2=self.config.attack_train_max_eps_l2,
                               n_classes=self.config.n_classes)
        x_adv = adversary.perturb(b, l).detach()
        assert not torch.isnan(x_adv).any().bool().item()
        requires_grad_(self.model, True)
        self.optimizer.zero_grad()

        if self.config.attack_percentage_clean != 0:
            n_clean = int(self.config.attack_percentage_clean * x_adv.shape[0])
            x_adv[:n_clean] = b[:n_clean]
        u.check_bounds(x_adv)
        return x_adv, l.detach()


class ActivationMatchingAdversarialTrainer(ActivationMatchingTrainer, AdversarialTrainer):
    def preprocess_data(self, b, l, epoch=None):
        x_adv, l = AdversarialTrainer.preprocess_data(self, b, l, epoch=None)
        b_new = torch.stack([b.to(u.dev()), x_adv], dim=1)
        clean_and_adv, l = ActivationMatchingTrainer.preprocess_data(self, b_new, l, epoch=None)
        return clean_and_adv, l


class AdversarialOrthogonalProjectionTrainer(AdversarialTrainer):
    def preprocess_data(self, b, l, epoch=None):
        b, l = b.to(u.dev()), l.to(u.dev())
        x_adv, l = super().preprocess_data(b, l, epoch=None)
        x_adv = orthogonal_projection(b, x_adv).detach()
        return x_adv, l


class ActivationMatchingAdversarialOrthogonalProjectionTrainer(ActivationMatchingTrainer,
                                                               AdversarialOrthogonalProjectionTrainer):
    def preprocess_data(self, b, l, epoch=None):
        b, l = b.cuda(), l.cuda()
        x_adv, l = AdversarialOrthogonalProjectionTrainer.preprocess_data(self, b, l, epoch=None)
        b_new = torch.stack([b, x_adv], dim=1)
        clean_and_adv, l = ActivationMatchingTrainer.preprocess_data(self, b_new, l)
        return clean_and_adv, l


class RedundancyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = None

    def forward(self, b):
        return self.model(b, return_individuals=True)

    def reset_counters(self):
        super().reset_counters()
        self.epoch_stats['abs_cosine_similarity'] = []
        self.epoch_stats['abs_scalar_product_similarity'] = []
        self.epoch_stats['similarity_measure'] = []
        self.epoch_stats[f'train_{self.config.similarity_measure}_exp_{self.config.projection_exponent}'] = []
        self.epoch_stats['gradient_magnitude'] = []

    def loss(self, outputs, l):
        bs, n_classes = l.shape[0], self.config.n_classes
        n_redundant = self.model.n_redundant

        _, individual_logits = outputs
        assert not torch.isnan(individual_logits).any().bool().item()
        assert individual_logits.shape == (l.shape[0], n_redundant, n_classes)  # shape convention

        if self.mask is not None:
            def modify_grads(x, inds):
                x[:, inds] = 0
                return x
            individual_logits.register_hook(lambda x: modify_grads(x, self.mask))

        l_ind = l[:, None].expand((bs, n_redundant)).flatten()    # same as n_redundant
        ce_ind = F.cross_entropy(individual_logits.view(-1, self.config.n_classes), l_ind)  # all in b dim
        loss = ce_ind

        # get gradients
        selected_logits = select_logits(individual_logits.view(bs*n_redundant, n_classes), l_ind,
                                        self.config.logits_for_similarity, n_classes).view(bs, n_redundant)
        sensitivity_vectors = get_grads_wrt_input(self.model, selected_logits, self.config.all_in_one_model,
                                                  create_graph=True).view(bs, n_redundant, -1)

        if self.config.cosine_only_for_top_k != 0:
            # only calculate the cosine similarity for the top k values and ingore the others
            k = self.config.cosine_only_for_top_k
            b_inds = torch.arange(bs).view(bs, 1).repeat(1, k * n_redundant).flatten()  # equivalent to tile
            n_redundant_inds = torch.arange(n_redundant).view(n_redundant, 1).repeat(1, k).flatten().repeat(bs)
            best_k = torch.topk(sensitivity_vectors, k, dim=-1).indices.flatten()
            assert len(b_inds) == len(n_redundant_inds) == len(best_k)

            tmp = torch.zeros_like(sensitivity_vectors)
            tmp[b_inds, n_redundant_inds, best_k] = sensitivity_vectors[[b_inds, n_redundant_inds, best_k]]
            sensitivity_vectors = tmp

        # enforce orthogonality of grads
        similarity_measure = calc_similarity_estimator(sensitivity_vectors,
                                                       similarity_measure=self.config.similarity_measure,
                                                       projection_exponent=self.config.projection_exponent)
        mask = torch.triu(torch.ones(bs, n_redundant, n_redundant), diagonal=1).type(torch.bool)
        assert mask.shape == similarity_measure.shape
        similarity_measure_mean = similarity_measure[mask].mean()  # get upper off diagonal elements

        loss += self.config.dissimilarity_weight * similarity_measure_mean
        if self.config.gradient_regularization_weight != 0:
            loss += self.config.gradient_regularization_weight * (sensitivity_vectors**2).mean(dim=0).sum()

        # for plotting
        selected_logits = select_logits(individual_logits.view(bs*n_redundant, n_classes), l_ind,
                                        'target', n_classes).view(bs, n_redundant)
        sensitivity_vectors = get_grads_wrt_input(self.model, selected_logits, self.config.all_in_one_model,
                                                  create_graph=False).detach().view(bs, n_redundant, -1)
        abs_cosine_similarity_plot = calc_similarity_estimator(sensitivity_vectors.detach()).cpu().abs()
        abs_scalar_product_similarity_plot = calc_similarity_estimator(sensitivity_vectors.detach(),
                                                                       similarity_measure='scalar_prod_abs').cpu().abs()
        self.epoch_stats['gradient_magnitude'].append(sensitivity_vectors.detach().abs().sum())
        self.epoch_stats['cross_entropy'].append(ce_ind.detach())
        self.epoch_stats['similarity_measure'].append(similarity_measure.detach())
        self.epoch_stats['abs_cosine_similarity'].append(abs_cosine_similarity_plot)
        self.epoch_stats['abs_scalar_product_similarity'].append(abs_scalar_product_similarity_plot)
        self.epoch_stats[f'train_{self.config.similarity_measure}_exp_{self.config.projection_exponent}'].append(
            similarity_measure.detach().cpu())
        return loss

    def track_statistics(self, b, l, outputs):
        logits, _ = outputs
        super().track_statistics(b, l, logits)

    def write_stats(self, writer, epoch):
        gradient_magnitude = torch.mean(torch.stack(self.epoch_stats['gradient_magnitude']))
        writer.add_scalar('train/gradient_magnitude', gradient_magnitude, epoch)
        super().write_stats(writer, epoch)

        # plot cosine sim
        n_redundant = self.model.n_redundant
        for name in ['abs_scalar_product_similarity', 'abs_cosine_similarity',
                     f'train_{self.config.similarity_measure}_exp_{self.config.projection_exponent}']:

            similarity = torch.cat(self.epoch_stats[name], dim=0).mean(dim=0)
            mask = torch.triu(torch.ones(n_redundant, n_redundant), diagonal=1).type(torch.bool)
            mean_sim = similarity[mask].mean()
            writer.add_scalar(f'train/{name}', mean_sim, epoch)

            # plot as matrix
            fig = u.plot_similarity_matrix(similarity, vmin=0, vmax=torch.max(similarity))
            writer.add_figure(f'train_sim_matrix/{name}', fig, epoch)


def calc_similarity_estimator(tensor, epsilon=1e-10, similarity_measure='cosine_similarity_abs',
                              projection_exponent=1.):
    """
    :param tensor: must be shape (bs, n_vectors, n_dims)
    :param epsilon: avoid division by zero
    :param similarity_measure: cosine_similarity_abs or scalar_prod_abs
    :param projection_exponent:
    :return: abs of similarity measure

    calculates a*b / (||a|| ||b||)
    """
    assert len(tensor.shape) == 3
    # = mat prod of column_vec * row_vec
    # detach to only make net_i orthogonal to net_<i and not the other way around (greedy)

    scalar_prod = torch.sum(tensor[:, :, None, :].detach()**projection_exponent * tensor[:, None, :, :], dim=3)
    if similarity_measure == 'scalar_prod_abs':
        return scalar_prod.abs()    # (bs, n_vecs, n_vecs)
    elif similarity_measure == 'cosine_similarity_abs':
        a_norm = torch.sqrt(torch.sum(tensor**2, dim=2))   # shape: (bs, n_vecs)
        norms = a_norm[:, :, None].detach() * a_norm[:, None, :]     # (bs, n_vecs, n_vecs)
        assert norms.shape == scalar_prod.shape
        return torch.abs(scalar_prod / (norms + epsilon))
    elif similarity_measure == 'random_vec':
        scalar_prod = torch.sum(torch.randn_like(tensor)[:, :, None, :].detach().abs() * tensor[:, None, :, :], dim=3)
        return scalar_prod.abs()
    else:
        print('similarity measure', similarity_measure)
        raise NotImplementedError()


def get_grads_wrt_input(model, selected_logits, all_in_one_model, create_graph=True):
    n_redundant = model.n_redundant
    if all_in_one_model:
        selected_logits = selected_logits.sum(dim=0)   # (bs, n_redundant) --> (n_redundant)
        grads_wrt_input = []
        for grad_loss_i in selected_logits:
            wrt = [model.cached_batch, *[model.layers[i] for i in [3, 4]]]
            # wrt = [self.model.cached_batch]
            grads = torch.autograd.grad(grad_loss_i, wrt, create_graph=create_graph, retain_graph=True)[:len(wrt)]
            grad = torch.cat([grad_i.flatten(1) for grad_i in grads], dim=1)
            grads_wrt_input.append(grad)

        grads_wrt_input = torch.stack(grads_wrt_input, dim=1)   # convention (bs, n_redundant, n_ch, n_x, n_y)
    else:
        assert model.cached_batches.shape[:2] == (selected_logits.shape[0], n_redundant)
        grads_wrt_input = torch.autograd.grad([selected_logits.sum()], model.cached_batches, create_graph=create_graph,
                                              retain_graph=True)[0]   # (bs, n_redundant, n_ch, n_x, n_y)
    return grads_wrt_input


def select_logits(logits, l, logits_for_similarity, n_classes):
    # enforce orthogonality of Jacobean with low cosine similarity
    assert l.shape[0] == logits.shape[0]
    bs = l.shape[0]
    x_inds = range(logits.shape[0])   # batch and individual nets in one dim
    if logits_for_similarity == 'target':
        selected_logits = logits[x_inds, l]
    elif logits_for_similarity == 'target_vs_best_other':
        best_other_logits, _ = u.get_best_non_target_logit(logits, l)
        selected_logits = logits[x_inds, l] - best_other_logits
    elif logits_for_similarity == 'target_vs_all_other':
        selected_logits = torch.zeros((bs, n_classes),
                                      dtype=torch.float32).to(u.dev())
        selected_logits[x_inds, l] = 2 * logits[x_inds, l]
        selected_logits -= logits  # equivalent to correct logit - wrong logits
        selected_logits = selected_logits.sum(dim=1)
    else:
        raise NotImplementedError(f'option for logits_for_similarity {logits_for_similarity} not implemented')
    assert selected_logits.shape == (bs, )
    return selected_logits
