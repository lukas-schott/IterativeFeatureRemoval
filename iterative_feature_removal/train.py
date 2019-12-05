import torch
from torch.nn import functional as F
from iterative_feature_removal import utils as u
from iterative_feature_removal.attacks import get_attack, orthogonal_projection
from torch import optim
from iterative_feature_removal.networks import requires_grad_


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
    def forward(self, b):
        return self.model(b, return_individuals=True)

    def reset_counters(self):
        super().reset_counters()
        self.epoch_stats['abs_cosine_similarity'] = []
        self.epoch_stats['gradient_magnitude'] = []

    def get_logits(self, individual_logits, n_redundant, bs, l_ind):
        # enforce orthogonality of Jacobean with low cosine similarity
        x_inds = range(individual_logits.shape[0] * n_redundant)   # batch and individual nets in one dim
        if self.config.all_logits:
            correct_logits = torch.zeros((n_redundant * bs, self.config.n_classes),
                                         dtype=torch.float32).to(u.dev())
            correct_logits[x_inds, l_ind] = 2 * individual_logits.reshape(-1, self.config.n_classes)[x_inds, l_ind]
            correct_logits -= individual_logits.reshape(-1, self.config.n_classes)  # correct logit - wrong logits
        else:
            correct_logits = individual_logits.reshape(-1, self.config.n_classes)[x_inds, l_ind]
        return correct_logits

    def get_grads_wrt_input(self, grad_loss, bs, n_redundant):
        if not self.config.all_in_one_model:
            assert self.model.cached_batches.shape[:2] == (bs, n_redundant)
        grads_wrt_input = torch.autograd.grad([grad_loss], self.model.cached_batches, create_graph=True,
                                              retain_graph=True)[0].abs()
        return grads_wrt_input

    def loss(self, outputs, l):
        _, individual_logits = outputs
        assert not torch.isnan(individual_logits).any().bool().item()
        bs = l.shape[0]
        n_redundant = self.model.n_redundant
        l_ind = l[:, None].expand((bs, n_redundant)).flatten()    # same as n_redundant
        ce_ind = F.cross_entropy(individual_logits.view(-1, self.config.n_classes), l_ind)  # all in b dim
        loss = ce_ind

        selected_logits = self.get_logits(individual_logits, n_redundant, bs, l_ind)
        grad_loss = torch.sum(selected_logits)
        # get abs cosine similarity
        grads_wrt_input = self.get_grads_wrt_input(grad_loss, bs, n_redundant)
        sensitivity_vectors = grads_wrt_input.view(bs, n_redundant, -1)

        if self.config.cosine_only_for_top_k != 0:
            # only calculate the cosine similarity for the top k values and ingore the others
            k = self.config.cosine_only_for_top_k
            b_inds = torch.arange(bs).view(bs, 1).repeat(1, k* n_redundant).flatten()  # equivalent to tile
            n_redundant_inds = torch.arange(n_redundant).view(n_redundant, 1).repeat(1, k).flatten().repeat(bs)
            best_k = torch.topk(sensitivity_vectors, k, dim=-1).indices.flatten()
            assert len(b_inds) == len(n_redundant_inds) == len(best_k)

            tmp = torch.zeros_like(sensitivity_vectors)
            tmp[b_inds, n_redundant_inds, best_k] = sensitivity_vectors[[b_inds, n_redundant_inds, best_k]]
            sensitivity_vectors = tmp

        abs_cosine_similarity = calc_abs_cosine_similarity(
            sensitivity_vectors.abs(), scalar_prod_as_similarity=self.config.scalar_prod_as_similarity)
        mask = torch.triu(torch.ones(bs, n_redundant, n_redundant),
                          diagonal=1).type(torch.bool)
        assert mask.shape == abs_cosine_similarity.shape
        abs_cosine_similarity = abs_cosine_similarity[mask]  # get off diagonal elements
        abs_cosine_similarity = abs_cosine_similarity.mean()

        loss += self.config.cosine_dissimilarity_weight * abs_cosine_similarity

        if self.config.gradient_regularization_weight != 0:
            loss += self.config.gradient_regularization_weight * (sensitivity_vectors**2).mean(dim=0).sum()


        self.epoch_stats['gradient_magnitude'].append(grads_wrt_input.abs().sum())
        self.epoch_stats['cross_entropy'].append(ce_ind.detach())
        self.epoch_stats['abs_cosine_similarity'].append(abs_cosine_similarity.detach())
        return loss

    def track_statistics(self, b, l, outputs):
        logits, _ = outputs
        super().track_statistics(b, l, logits)

    def write_stats(self, writer, epoch):
        abs_cosine_similarities = torch.mean(torch.stack(self.epoch_stats['abs_cosine_similarity']))
        print('cosine sim',  torch.mean(torch.stack(self.epoch_stats['abs_cosine_similarity'])))
        writer.add_scalar('train/abs_cosine_similarity', abs_cosine_similarities, epoch)
        gradient_magnitude = torch.mean(torch.stack(self.epoch_stats['gradient_magnitude']))
        writer.add_scalar('train/gradient_magnitude', gradient_magnitude, epoch)
        super().write_stats(writer, epoch)


def calc_abs_cosine_similarity(tensor, epsilon=1e-10, scalar_prod_as_similarity=False):
    """
    :param tensor: must be shape (bs, n_vectors, n_dims)
    :param epsilon: avoid division by zero
    :return: abs of cosine similarity

    calculates a*b / (||a|| ||b||)
    """
    assert len(tensor.shape) == 3
    # = mat prod of column_vec * row_vec
    # detach to only make net_i orthogonal to net_<i and not the other way around (greedy)

    scalar_prod = torch.sum(tensor[:, :, None, :].detach() * tensor[:, None, :, :], dim=3)  # (bs, n_vecs, n_vecs)
    if scalar_prod_as_similarity:
        return scalar_prod
    a_norm = torch.sqrt(torch.sum(tensor**2, dim=2))   # shape: (bs, n_vecs)

    # b_norm = (tensor[:, :, None, :].detach()**2 * tensor[:, None, :, :]**2).sum(dim=3).sqrt()     # (bs, n_vecs, n_vecs)
    b_norm = a_norm[:, None, :]  # standard

    # tmp = a_norm[:, None, :]
    norms = a_norm[:, :, None].detach() * b_norm     # (bs, n_vecs, n_vecs)

    assert norms.shape == scalar_prod.shape
    return torch.abs(scalar_prod / (norms + epsilon))



