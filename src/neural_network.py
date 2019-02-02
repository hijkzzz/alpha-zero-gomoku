# -*- coding: utf-8 -*-
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


class NeuralNetWork(torch.jit.ScriptModule):
    """Policy and Value Network
    """
    __constants__ = ['conv1', 'conv2', 'conv3', 'pi_conv', 'pi_fc', 'v_conv', 'v_fc1', 'v_fc2']

    def __init__(self, num_channels, n, action_size):
        super(NeuralNetWork, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, num_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1), nn.ReLU())

        self.pi_conv = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1, padding=0), nn.ReLU())
        self.pi_fc = nn.Sequential(nn.Linear(4 * n ** 2, action_size), nn.ReLU(), nn.LogSoftmax(dim=1))

        self.v_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, padding=0), nn.ReLU())
        self.v_fc1 = nn.Sequential(nn.Linear(2 * n ** 2, 64), nn.ReLU())
        self.v_fc2 = nn.Sequential(nn.Linear(64, 1), nn.Tanh())

    @torch.jit.script_method
    def forward(self, states):
        out = self.conv1(states)
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

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))

        return value_loss + policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, lr, l2, kl_targ, epochs, num_channels, n, action_size):
        """ init
        """
        if not torch.cuda.is_available():
            print("cuda is unavailable")
            exit(1)

        self.lr = lr
        self.l2 = l2
        self.kl_targ = kl_targ
        self.epochs = epochs
        self.num_channels = num_channels
        self.n = n

        self.neural_network = NeuralNetWork(num_channels, n, action_size)
        self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.alpha_loss = AlphaLoss()

    def train(self, example_batch):
        """train neural network
        """
        # extract train data
        board_batch, last_action_batch, cur_player_batch, p_batch, v_batch = list(zip(*example_batch))

        state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
        p_batch = torch.Tensor(p_batch).cuda()
        v_batch = torch.Tensor(v_batch).cuda().unsqueeze(1)

        # for calculating KL divergence
        old_p, old_v = self._infer(state_batch)

        for epoch in range(self.epochs):
            self.neural_network.train()

            # zero the parameter gradients
            self.set_learning_rate(self.lr)
            self.optim.zero_grad()

            # forward + backward + optimize
            log_ps, vs = self.neural_network(state_batch)
            loss = self.alpha_loss(log_ps, vs, p_batch, v_batch)
            loss.backward()

            self.optim.step()

            # calculate KL divergence
            new_p, _ = self._infer(state_batch)

            kl = np.mean(np.sum(old_p * (
                np.log(old_p + 1e-10) - np.log(new_p + 1e-10)),
                axis=1)
            )

            # early stopping if D_KL diverges badly
            if kl > self.kl_targ:
                break

        print("LOSS :: {},  KL :: {}".format(loss.item(), kl))

    def infer(self, feature_batch):
        """predict p and v by raw input
           return numpy
        """
        board_batch, last_action_batch, cur_player_batch = list(zip(*feature_batch))
        states = self._data_convert(board_batch, last_action_batch, cur_player_batch)

        self.neural_network.eval()
        log_ps, vs = self.neural_network(states)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _infer(self, state_batch):
        """predict p and v by state
           return numpy object
        """

        self.neural_network.eval()
        log_ps, vs = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """convert data format
           return tensor
        """
        n = self.n

        board_batch = torch.Tensor(board_batch).cuda().unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        state2 = torch.zeros((len(last_action_batch), 1, n, n)).cuda().float()
        state3 = torch.ones((len(cur_player_batch), 1, n, n)).cuda().float()

        for i in range(len(cur_player_batch)):
            state3[i][0] *= cur_player_batch[i]

            last_action = last_action_batch[i]
            if last_action != -1:
                x, y = last_action // self.n, last_action % self.n
                state2[i][0][x][y] = 1

        return torch.cat((state0, state1, state2, state3), dim=1)

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

        # output for c++
        filepath = os.path.join(folder, filename + '.pt')
        self.neural_network.eval()
        self.neural_network.save(filepath)
