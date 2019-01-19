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
            nn.Conv2d(4, args.num_channels, kernel_size=3, padding=1), nn.ReLU())
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

    def __init__(self, args):
        super(AlphaLoss, self).__init__()
        self.args = args

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))

        return value_loss + policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, args):
        """args: lr, l2, batch_size, dropout
        """

        self.args = args
        self.cuda = torch.cuda.is_available()
        self.neural_network = NeuralNetWork(args)

        if self.cuda:
            self.neural_network.cuda()
            print("CUDA ON")

        self.optim = Adam(self.neural_network.parameters(), lr=args.lr, weight_decay=args.l2)
        self.alpha_loss = AlphaLoss(self.args)

    def train(self, example_batch):
        """train neural network
        """

        # extract train data
        board_batch, p_batch, v_batch, last_action_batch, cur_player_batch = list(zip(*[example for example in example_batch]))

        state_batch = self.__data_convert(board_batch, last_action_batch, cur_player_batch)
        p_batch = torch.Tensor(p_batch)
        v_batch = torch.Tensor(v_batch).unsqueeze(1)

        if self.cuda:
            state_batch = state_batch.cuda()
            p_batch = p_batch.cuda()
            v_batch = v_batch.cuda()

        # for calculating KL divergence
        old_p, old_v = self.__infer(state_batch)

        for epoch in range(self.args.epochs):
            self.neural_network.train()

            # zero the parameter gradients
            self.set_learning_rate(self.args.lr)
            self.optim.zero_grad()

            # forward + backward + optimize
            log_ps, vs = self.neural_network(state_batch)
            loss = self.alpha_loss(log_ps, vs, p_batch, v_batch)
            loss.backward()

            self.optim.step()

            # calculate KL divergence
            new_pi, new_v = self.__infer(state_batch)

            kl = np.mean(np.sum(old_p * (
                np.log(old_v + 1e-10) - np.log(new_pi + 1e-10)),
                axis=1)
            )

            # early stopping if D_KL diverges badly
            if kl > self.args.kl_targ * 4:
                break

        print("LOSS :: {}, LR :: {}, KL :: {}".format(loss.item(), self.args.lr, kl))

        # adaptively adjust the learning rate
        if kl > self.args.kl_targ * 2 and self.args.lr > 0.001:
            self.args.lr /= 1.5
        elif kl < self.args.kl_targ / 2 and self.args.lr < 0.1:
            self.args.lr *= 1.5


    def infer(self, board_batch, last_action_batch, cur_player_batch):
        """predict p and v by raw input
           return list
        """

        states = self.__data_convert(board_batch, last_action_batch, cur_player_batch)
        if self.cuda:
            states = states.cuda()

        self.neural_network.eval()
        log_ps, vs  = self.neural_network(states)

        res = (np.exp(log_ps.cpu().detach().numpy()).tolist(), vs.cpu().detach().numpy().tolist())

        return res

    def __infer(self, state_batch):
        """predict p and v by state
           return numpy object
        """

        self.neural_network.eval()
        log_ps, vs  = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def __data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """convert data format
           return tensor
        """

        board_batch = torch.Tensor(board_batch).unsqueeze(1)

        player1_batch0 = (board_batch > 0).float()
        plater_1_batch0 = (board_batch < 0).float()
        last_action_batch0 = torch.zeros((len(last_action_batch), 1, self.args.n, self.args.n)).float()
        cur_player_batch0 = torch.ones((len(cur_player_batch), 1, self.args.n, self.args.n)).float()

        for i in range(len(cur_player_batch0)):
            cur_player_batch[i][0] *= cur_player_batch[i]

            last_action = last_action_batch[i]
            if not last_action is None:
                x, y = last_action
                last_action_batch0[i][0][x][y] = 1

        # DEBUG
        if self.args.debug == True:
            print(torch.cat((player1_batch0, plater_1_batch0, last_action_batch0, cur_player_batch0), dim=1))

        return torch.cat((player1_batch0, plater_1_batch0, last_action_batch0, cur_player_batch0), dim=1)

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
