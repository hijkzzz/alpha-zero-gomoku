# -*- coding: utf-8 -*-
import numpy as np
import sys
sys.path.append('../src')

import neural_network
import torch


if __name__ == "__main__":
    lr = 0.002
    l2 = 0.0002
    epochs = 5
    kl_targ = 0.04
    num_channels = 128
    n = 5
    action_size = n ** 2

    policy_value_net = neural_network.NeuralNetWorkWrapper(lr, l2, kl_targ, epochs, num_channels, n, action_size)

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

    log_ps, vs = policy_value_net.neural_network(state_batch.cuda())
    print('log_p, v \n', np.exp(log_ps.cpu().detach().numpy()), vs.cpu())

    loss = policy_value_net.alpha_loss(log_ps, vs, p_batch, v_batch.unsqueeze(1))

    print('loss \n', loss.cpu())

    # test train
    example_batch = list(zip(board_batch, last_action_batch, cur_player_batch,
                             p_batch.cpu().numpy().tolist(), v_batch.cpu().numpy().tolist()))
    print('train\n', example_batch)

    policy_value_net.train(example_batch)

    # test infer
    print('infer \n', policy_value_net.infer(list(zip(board_batch, last_action_batch, cur_player_batch))))

    #save model
    neural_network.NeuralNetWorkWrapper(lr, l2, kl_targ, epochs, 256, 10, 100).save_model()
