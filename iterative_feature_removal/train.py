import torch
from iterative_feature_removal import utils as u
from torch import optim


def get_trainer(config):
    if config.training_mode == 'normal' or config.training_mode == 'overwrite':
        print('vanilla trainer')
        return Trainer
    elif config.training_mode == 'append_dataset':
        print('siamese trainer')
        return SiameseTrainer
    else:
        print('mode', config.training_mode)
        raise NotImplementedError


class Trainer:
    def __init__(self, model: torch.nn.Module, data_loader,
                 optimizer: optim.Adam, class_loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        self.class_loss_fct = class_loss_fct
        self.n_correct = 0
        self.optimizer = optimizer
        self.model = model
        self.data_loader = data_loader

    def train_epoch(self):
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
        return self.class_loss_fct(logits, target=l)

    def preprocess_data(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        return b, l

    def train_iter(self, b, l):
        b, l = self.preprocess_data(b, l)
        output = self.forward(b)
        loss = self.loss(output, l)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.track_statistics(b, l, output)

    def track_statistics(self, b, l, logits):
        with torch.no_grad():
            self.n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            self.n_total += b.shape[0]


class SiameseTrainer(Trainer):
    def forward(self, b):
        return self.model(b, return_all=True)

    def preprocess_data(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        b = b.view(b.shape[0] * b.shape[1], *b.shape[2:])
        l = torch.stack([l, l], dim=1).flatten()
        return b, l

    def loss(self, output, l):
        logits, activations = output
        class_loss = self.class_loss_fct(logits, l)
        activation_loss = 0
        for activation in activations:
            activation_loss += torch.mean((activation[::2] - activation[1::2])**2)
        loss = class_loss + activation_loss
        return loss

    def track_statistics(self, b, l, outputs):
        logits, _ = outputs
        super().track_statistics(b, l, logits)


# class AdversarialTrainer(Trainer):
#     def


