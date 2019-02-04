# -*- coding: utf-8 -*-
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out


class NeuralNetWork(nn.Module):
    """Policy and Value Network
    """

    def __init__(self, num_channels, n, action_size):
        super(NeuralNetWork, self).__init__()

        # residual block
        self.res1 = ResidualBlock(4, num_channels)
        self.res2 = ResidualBlock(num_channels, num_channels)
        self.res3 = ResidualBlock(num_channels, num_channels)
        self.res4 = ResidualBlock(num_channels, num_channels)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU(inplace=True)

        self.p_fc = nn.Linear(4 * n ** 2, action_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # value head
        self.v_conv = nn.Conv2d(num_channels, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(num_features=2)

        self.v_fc1 = nn.Linear(2 * n ** 2, 128)
        self.v_fc2 = nn.Linear(128, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res1(inputs)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        # policy head
        p = self.p_conv(out)
        p = self.p_bn(p)
        p = self.relu(p)

        p = self.p_fc(p.view(p.size(0), -1))
        p = self.log_softmax(p)

        # value head
        v = self.v_conv(out)
        v = self.v_bn(v)
        v = self.relu(v)

        v = self.v_fc1(v.view(v.size(0), -1))
        v = self.relu(v)
        v = self.v_fc2(v)
        v = self.tanh(v)

        return p, v


class AlphaLoss(nn.Module):
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

            entropy = -np.mean(
                np.sum(new_p * np.log(new_p + 1e-10), axis=1)
            )

            # early stopping if D_KL diverges badly
            if kl > self.kl_targ:
                break

        print("LOSS :: {}, ENTROPY :: {}, KL :: {}".format(loss.item(), entropy, kl))

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

    def load_model(self, folder="models", filename="checkpoint"):
        """load model from file
        """

        filepath = os.path.join(folder, filename)
        self.neural_network.load_state_dict(torch.load(filepath))

    def save_model(self, folder="models", filename="checkpoint"):
        """save model to file
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.neural_network.state_dict(), filepath)

        # output for libtorch
        filepath = os.path.join(folder, filename + '.pt')

        self.neural_network.eval()
        example = torch.rand(1, 4, self.n, self.n).cuda()
        traced_script_module = torch.jit.trace(self.neural_network, example)
        traced_script_module.save(filepath)
