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
            nn.Conv2d(1, args.num_channel, kernel_size=3, padding=2), nn.BatchNorm2d(args.num_channel), nn.ReLU())
        # n
        self.conv2 = nn.Sequential(
            nn.Conv2d(args.num_channel, args.num_channel, kernel_size=3, padding=2), nn.BatchNorm2d(args.num_channel), nn.ReLU())
        # n
        self.conv3 = nn.Sequential(
            nn.Conv2d(args.num_channel, args.num_channel, kernel_size=3, padding=0), nn.BatchNorm2d(args.num_channel), nn.ReLU())
        # n - 2
        self.conv4 = nn.Sequential(
            nn.Conv2d(args.num_channel, args.num_channel, kernel_size=3, padding=0), nn.BatchNorm2d(args.num_channel), nn.ReLU())
        # n - 4
        self.fc1 = nn.Sequential(nn.Linear(args.num_channel * (args.n - 4) ** 2, 1024), nn.ReLU(), nn.Dropout(p=args.dropout))
        self.fc2 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(p=args.dropout))

        self.v = nn.Sequential(nn.Linear(1024, 1), nn.Tanh())
        self.pi = nn.Sequential(nn.Linear(512, args.action_size), nn.Softmax())

    def forward(self, boards):
        out = self.conv1(boards)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.size(0), -1)
        fc1 = self.fc1(out)
        fc2 = self.fc2(fc1)

        v =  self.v(fc1)
        pi = self.pi(fc2)

        return v, pi


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, neural_network, args, cuda=False):
        """args: lr, l2, batch_size, dropout
        """

        self.neural_network = neural_network
        self.cuda = cuda
        self.args = args

        if self.cuda:
            self.neural_network.cuda()

        self.optim = Adam(self.neural_network.parameters(), lr=args.lr, weight_decay=args.l2)


    def train(self, examples):
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))

            batch_idx = 0
            while batch_idx < int(len(examples) / args.batch_size):
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards, pis, vs = Variable(torch.cat(boards)), Variable(torch.cat(pis)), Variable(torch.cat(vs))
                if self.cuda:
                    boards, pis, vs = boards.cuda(), pis.cuda(), vs.cuda()

                boards, pis, vs = boards.unsqueeze(3), pis.unsqueeze(1), vs.unsqueeze(1)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.neural_network(boards)
                loss = 
                loss.backward()
                optimizer.step()

                



    def infer(self, board):


    def load_model(self, filename="checkpoint", folder="models"):
        filepath = os.path.join(folder, filename)
        self.neural_network.load_state_dict(torch.load(filepath))

        
    def save_model(self, filename="checkpoint", folder="models"):
        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        torch.save(self.actor_target.state_dict(), filepath)
        

