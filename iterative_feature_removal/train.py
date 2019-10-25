import torch
from iterative_feature_removal import utils as u
from iterative_feature_removal.attacks import get_attack
from torch import optim


def get_trainer(config):
    if config.training_mode == 'normal' or config.training_mode == 'overwrite':
        print('vanilla trainer')
        return Trainer
    elif config.training_mode == 'append_dataset':
        print('siamese trainer')
        return SiameseTrainer
    elif config.training_mode == 'adversarial_training':
        return AdversarialTrainer
    elif config.training_mode == 'siamese_adversarial_training':
        return AdversarialTrainer
    else:
        print('mode', config.training_mode)
        raise NotImplementedError


def get_optimizer(config, model):
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
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

    def train_epoch(self):
        self.optimizer.zero_grad()
        self.model.train()
        self.optimizer.zero_grad()

        self.n_correct, self.n_total = 0., 0.
        for i, (b, l) in enumerate(self.data_loader):
            self.train_iter(b, l)

        accuracy = self.n_correct / self.n_total
        return accuracy

    def forward(self, b):
        return self.model(b)

    def loss(self, logits, l):
        assert not torch.isnan(logits).any().bool().item()
        return self.class_loss_fct(logits, target=l)

    def preprocess_data(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        return b, l

    def train_iter(self, b, l):
        b, l = self.preprocess_data(b, l)
        output = self.forward(b)
        loss = self.loss(output, l)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.track_statistics(b, l, output)

    def track_statistics(self, b, l, logits):
        with torch.no_grad():
            self.n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            self.n_total += b.shape[0]


class SiameseTrainer(Trainer):
    def forward(self, b):
        return self.model(b, return_activations=True)

    def preprocess_data(self, b, l):
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
    def preprocess_data(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        logits = self.model(b)
        assert not torch.isnan(logits).any().bool().item()

        adversary = get_attack(self.model, self.config.lp_metric, self.config.adv_train_attack_name,
                               self.config.adv_attack_iter,
                               l2_step_size=self.config.adv_l2_step_size,
                               max_eps_l2=self.config.adv_train_epsilon,
                               n_classes=self.config.n_classes)
        x_adv = adversary.perturb(b, l).detach()
        self.optimizer.zero_grad()

        assert not torch.isnan(x_adv).any().bool().item()
        u.check_bounds(x_adv)
        return x_adv, l.detach()


class SiameseAdversarialTrainer(SiameseTrainer, AdversarialTrainer):
    def preprocess_data(self, b, l):
        x_adv, l = AdversarialTrainer.preprocess_data(self, b, l)
        b_new = torch.stack([b, x_adv], dim=1)
        clean_and_adv, l = SiameseTrainer.preprocess_data(self, b_new, l)
        return clean_and_adv, l