# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')
sys.path.append('../build')

import torch
from library import Gomoku
import neural_network
import numpy as np


def tuple_2d_to_numpy_2d(tuple_2d):
    # help function
    # convert type
    res = [None] * len(tuple_2d)
    for i, tuple_1d in enumerate(tuple_2d):
        res[i] = list(tuple_1d)
    return np.array(res)


if __name__ == "__main__":
    lr = 0.002
    l2 = 0.0002
    epochs = 5
    num_layers = 4
    num_channels = 128
    n = 5
    action_size = n ** 2

    policy_value_net = neural_network.NeuralNetWorkWrapper(lr, l2, num_layers, num_channels, n, action_size)

    # test data convert
    board_batch = [[[1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1]],
                   [[1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1], [1, 0, -1, 0, -1]]]
    last_action_batch = [-1, 0]
    cur_player_batch = [1, -1]

    state_batch = policy_value_net._data_convert(board_batch, last_action_batch, cur_player_batch)
    print('state \n', state_batch)

    # test loss
    p_batch = torch.Tensor([[1 / 25 for _ in range(25)], [1 / 25 for _ in range(25)]]).cuda()
    v_batch = torch.Tensor([0.5, 0.5]).cuda()

    log_p, v = policy_value_net.neural_network(state_batch.cuda())
    print('p, v \n', np.exp(log_p.cpu().detach().numpy()), v.cpu())

    loss = policy_value_net.alpha_loss(log_p, v, p_batch, v_batch.unsqueeze(1))

    print('loss \n', loss.cpu())

    # test train
    example_batch = list(zip(board_batch, last_action_batch, cur_player_batch,
                             p_batch.cpu().numpy().tolist(), v_batch.cpu().numpy().tolist()))
    print('train\n', example_batch)

    policy_value_net.train(example_batch, len(example_batch), epochs)

    # test infer
    print('infer \n', policy_value_net.infer(list(zip(board_batch, last_action_batch, cur_player_batch))))

    # test libtorch
    nn = neural_network.NeuralNetWorkWrapper(lr, l2, 4, 256, 15, 225, True, True)
    nn.save_model(folder="models", filename="checkpoint")
    # nn.load_model(folder="models", filename="checkpoint")

    gomoku = Gomoku(15, 5, 1)
    gomoku.execute_move(3)
    gomoku.execute_move(4)
    gomoku.execute_move(6)
    gomoku.execute_move(23)
    gomoku.execute_move(8)
    gomoku.execute_move(9)
    gomoku.execute_move(78)
    gomoku.execute_move(0)
    gomoku.execute_move(17)
    gomoku.execute_move(7)
    gomoku.execute_move(19)
    gomoku.execute_move(67)
    gomoku.execute_move(60)
    gomoku.execute_move(14)
    gomoku.execute_move(11)
    gomoku.execute_move(2)
    gomoku.execute_move(99)
    gomoku.execute_move(10)
    gomoku.execute_move(1)
    gomoku.execute_move(5)
    gomoku.execute_move(18)
    gomoku.execute_move(12)
    gomoku.execute_move(15)

    feature_batch = [(tuple_2d_to_numpy_2d(gomoku.get_board()), gomoku.get_last_move(), gomoku.get_current_color())]
    print(feature_batch)
    print(nn.infer(feature_batch))

    gomoku.execute_move(24)
    feature_batch = [(tuple_2d_to_numpy_2d(gomoku.get_board()), gomoku.get_last_move(), gomoku.get_current_color())]
    print(feature_batch)
    print(nn.infer(feature_batch))

