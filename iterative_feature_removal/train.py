import torch
from iterative_feature_removal import utils as u
from torch import optim


def get_trainer(config):
    if config.mode == 'overwrite_dataset':
        return Trainer
    elif config.mode == 'append_dataset':
        return SiameseTrainer


class Trainer:
    def __init__(self, model: torch.nn.Module, data_loader,
                 optimizer: optim.Adam, class_loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        print('vanilla trainer')
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

    def train_iter(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        logits = self.forward(b)

        loss = self.loss(self, logits, l)

        self.backward(loss)

        with torch.no_grad():
            self.n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            self.n_total += b.shape[0]

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


class SiameseTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('siamese trainer')

    def forward(self, b):
        return self.model(b, return_all=True)

    def train_iter(self, b, l):
        u.check_bounds(b)
        b, l = b.to(u.dev()), l.to(u.dev())
        b = b.view(b.shape[0] * b.shape[1], *b.shape[2:])
        l = torch.stack([l, l], dim=1).flatten()
        logits, activations = self.forward(b)
        loss = self.loss(logits, l, activations)

        self.backward(loss)

        with torch.no_grad():
            self.n_correct += float(torch.sum(torch.argmax(logits, dim=1) == l))
            self.n_total += b.shape[0]

    def loss(self, logits, l, activations):
        class_loss = self.class_loss_fct(logits, l)
        activation_loss = 0
        for activation in activations:
            activation_loss += torch.mean((activation[::2] - activation[1::2])**2)
        loss = class_loss + activation_loss
        return loss