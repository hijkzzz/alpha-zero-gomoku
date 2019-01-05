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

        return [v, pi]


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

        for epoch in range(self.args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            batch_idx = 0
            while batch_idx < int(len(examples) / self.args.batch_size):
                batch_idx += 1
                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards, pis, vs = torch.Tensor(boards).unsqueeze(1), \
                                torch.Tensor(pis), \
                                torch.Tensor(vs)

                if self.cuda:
                    boards, pis, vs = boards.cuda(), pis.cuda(), vs.cuda()

                # zero the parameter gradients
                self.optim.zero_grad()

                # forward + backward + optimize
                output_vs, output_pis = self.neural_network(boards)
                loss = alpha_loss(output_pis, output_vs, pis, vs)
                loss.backward()

                self.optim.step()

                print("BATCH ::: {}, LOSS ::: {}".format(batch_idx, loss.item()))

    def infer(self, board):
        """predict v and pi
        """

        self.neural_network.eval()

        boards = torch.Tensor(board).unsqueeze(0).unsqueeze(1)

        if self.cuda:
            boards = boards.cuda()

        output_vs, output_pis = self.neural_network(boards)

        return [output_pis[0].cpu().detach().numpy(), output_vs[0].cpu().detach().numpy()]

    
    def set_learning_rate(self, lr):
        """set learning rate
        """

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr


    def load_model(self, filename="checkpoint", folder="models"):
        """load model from file
        """

        filepath = os.path.join(folder, filename)
        if os.path.exists(filepath):
            self.neural_network.load_state_dict(torch.load(filepath))


    def save_model(self, filename="checkpoint", folder="models"):
        """save model to file
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.neural_network.state_dict(), filepath)
