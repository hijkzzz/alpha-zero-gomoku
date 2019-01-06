# -*- coding: utf-8 -*-
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class NeuralNetWork(nn.Module):
    """Policy and Value Network
    """

    def __init__(self, args):
        super(NeuralNetWork, self).__init__()
        # n
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, args.num_channels, kernel_size=3, padding=1), nn.ReLU())
        # n
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, padding=1), nn.ReLU())
        # n
        self.conv3 = nn.Sequential(
            nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, padding=1), nn.ReLU())

        self.pi_conv = nn.Sequential(
            nn.Conv2d(args.num_channels, 4, kernel_size=1, padding=0), nn.ReLU())
        self.pi_fc = nn.Sequential(nn.Linear(4 * args.n ** 2, args.action_size), nn.ReLU(), nn.LogSoftmax(dim=1))

        self.v_conv = nn.Sequential(
            nn.Conv2d(args.num_channels, 2, kernel_size=1, padding=0), nn.ReLU())
        self.v_fc1 = nn.Sequential(nn.Linear(2 * args.n ** 2, 64), nn.ReLU())
        self.v_fc2 = nn.Sequential(nn.Linear(64, 1), nn.Tanh())

    def forward(self, boards):
        out = self.conv1(boards)
        out = self.conv2(out)
        out = self.conv3(out)

        pi = self.pi_conv(out)
        pi = self.pi_fc(pi.view(pi.size(0), -1))

        v = self.v_conv(out)
        v = self.v_fc1(v.view(v.size(0), -1))
        v = self.v_fc2(v)

        return pi, v


class AlphaLoss(torch.nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, log_pis, vs, target_pis, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_pis * log_pis, 1))
        return value_loss + policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, neural_network, args):
        """args: lr, l2, batch_size, dropout
        """

        self.neural_network = neural_network
        self.cuda = torch.cuda.is_available()
        self.args = args

        if self.cuda:
            self.neural_network.cuda()
            print("CUDA ON")

        self.optim = Adam(self.neural_network.parameters(), lr=args.lr, weight_decay=args.l2)

    def train(self, examples):
        """train neural network
        """

        self.neural_network.train()
        alpha_loss = AlphaLoss()

        # prepare train data
        board_batch, pi_batch, vs_batch = list(zip(*[example for example in examples]))
        board_batch, pi_batch, vs_batch = torch.Tensor(board_batch).unsqueeze(1), \
            torch.Tensor(pi_batch), \
            torch.Tensor(vs_batch)

        if self.cuda:
            board_batch, pi_batch, vs_batch = board_batch.cuda(), pi_batch.cuda(), vs_batch.cuda()

        old_pi, old_v = self.infer(board_batch)

        for epoch in range(self.args.epochs):
            # zero the parameter gradients
            self.optim.zero_grad()
            self.set_learning_rate(self.args.lr)

            # forward + backward + optimize
            vs, log_pis = self.neural_network(board_batch)
            loss = alpha_loss(log_pis, vs, pi_batch, vs_batch)
            loss.backward()

            self.optim.step()

            # calculate KL
            new_pi, new_v = self.infer(board_batch)

            kl = np.mean(np.sum(old_pi * (
                np.log(old_pi + 1e-10) - np.log(new_pi + 1e-10)),
                axis=1)
            )

            # early stopping if D_KL diverges badly
            if kl > self.args.kl_targ * 4:  
                break

            # adaptively adjust the learning rate
            if kl > self.args.kl_targ * 2 and self.args.lr > 0.1:
                self.args.lr /= 1.5
            elif kl < self.args.kl_targ / 2 and self.args.lr < 10:
                self.args.lr *= 1.5

            print("EPOCH :: {}, LOSS :: {}, LR :: {}, KL :: {}".format(epoch + 1, loss.item(), self.args.lr, kl))

    def infer(self, board, data_format='MCTS'):
        """predict pi and v
        """

        self.neural_network.eval()

        if data_format == 'MCTS':
            boards = torch.Tensor(board).unsqueeze(0).unsqueeze(1)

            if self.cuda:
                boards = boards.cuda()

        log_pis, vs  = self.neural_network(boards)
        return np.exp(log_pis.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def set_learning_rate(self, lr):
        """set learning rate
        """

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def load_model(self, filename="checkpoint", folder="models"):
        """load model from file
        """

        filepath = os.path.join(folder, filename)
        self.neural_network.load_state_dict(torch.load(filepath))

    def save_model(self, filename="checkpoint", folder="models"):
        """save model to file
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.neural_network.state_dict(), filepath)
