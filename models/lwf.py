# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from cProfile import label
import torch
from datasets import get_dataset
from torch.optim import SGD
from utils.args import *
import torch.optim as optim
from models.utils.continual_model import ContinualModel
from argparse import ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Learning without Forgetting.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Penalty weight.')
    parser.add_argument('--softmax_temp', type=float, default=2,
                        help='Temperature of the softmax function.')
    return parser


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class Lwf(ContinualModel):
    NAME = 'lwf'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Lwf, self).__init__(backbone, loss, args, transform)
        self.old_net = None
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)
        self.dataset = get_dataset(args)
        self.current_task = 0
        self.cpt = get_dataset(args).N_CLASSES_PER_TASK
        nc = get_dataset(args).N_TASKS * self.cpt
        self.eye = torch.tril(torch.ones((nc, nc))).bool().to(self.device)

    def begin_task(self, train_loader):
        self.net.eval()
        if self.current_task > 0:
            # warm-up
            # opt = SGD(self.net.classifier.parameters(), lr=self.args.lr)
            opt = optim.Adam(self.net.parameters(), lr=self.args.lr)
            for epoch in range(self.args.n_epochs):
                for i, data in enumerate(train_loader):
                    # inputs, labels, not_aug_inputs = data
                    inputs0, inputs1, labels = data
                    inputs0, inputs1, labels = inputs0.to(self.device), inputs1.to(self.device), labels.to(self.device)
                    # inputs, labels = inputs.to(self.device), labels.to(self.device)
                    opt.zero_grad()
                    # with torch.no_grad():
                        # feats = self.net([inputs0,inputs1], returnt='features')
                    outputs= self.net([inputs0,inputs1])
                    logits, _,_,_,_ = outputs
                    mask = self.eye[(self.current_task + 1) * self.cpt - 1] ^ self.eye[self.current_task * self.cpt - 1]
                    # outputs = self.net(feats)[:, mask]
                    logits_masked= logits[:, mask]
                    loss = self.loss(logits_masked, labels - self.current_task * self.cpt)
                    loss.backward()
                    opt.step()

            logits = []
            with torch.no_grad():
                # import ipdb;ipdb.set_trace()
                # for i in range(0, dataset.train_loader.dataset.data.shape[0], self.args.batch_size):
                for i in range(0, len(train_loader.dataset), self.args.batch_size):
                    batch = [train_loader.dataset.__getitem__(j) 
                    for j in range(i, min(i + self.args.batch_size, len(train_loader.dataset)))]

                    inputs0 = torch.stack([item[0] for item in batch])  # Extract inputs0
                    inputs1 = torch.stack([item[1] for item in batch])  # Extract inputs1
                    inputs0, inputs1 = inputs0.to(self.device), inputs1.to(self.device)
                    inputs0=inputs0.squeeze(0)
                    inputs1=inputs1.squeeze(0)
                    # inputs = torch.stack([train_loader.dataset.__getitem__(j)[0]
                    #                       for j in range(i, min(i + self.args.batch_size,
                    #                                      len(train_loader.dataset)))])[0]
                    log = self.net([inputs0,inputs1])[0].cpu()
                    logits.append(log)
            setattr(train_loader.dataset, 'logits', torch.cat(logits))
        self.net.train()

        self.current_task += 1

    def observe(self, inputs0, inputs1, labels, not_aug_inputs=None, logits=None):
        self.opt.zero_grad()
        outputs = self.net([inputs0,inputs1])
        logits, _,_,_,_ = outputs
        # import ipdb;ipdb.set_trace()

        mask = self.eye[self.current_task * self.cpt - 1]
        loss = self.loss(outputs[0][:, mask], labels)
        if logits is not None:
            mask = self.eye[(self.current_task - 1) * self.cpt - 1]
            loss += self.args.alpha * modified_kl_div(smooth(self.soft(logits[:, mask]).to(self.device), 2, 1),
                                                      smooth(self.soft(logits[:, mask]), 2, 1))

        loss.backward()
        self.opt.step()

        return loss.item()
