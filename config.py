config = {
    # gomoku
    'n': 15,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts
    'mcts_use_gpu' : True,                      # mcts use cuda
    'thread_pool_size': 4,                      # mcts threads number
    'num_mcts_sims': 1600,                      # mcts simulation times
    'c_puct': 1.5,                                # puct coeff
    'c_virtual_loss': 1,                        # virtual loss coeff

    # neural_network
    'nn_use_gpu' : True,                        # neural network use cuda
    'lr': 0.001,                                # learning rate
    'l2': 0.0001,                               # L2
    'num_channels': 256,                        # convolution neural network channel size
    'epochs': 2,                                # train epochs
    'batch_size': 512,                          # batch size

    # train
    'num_iters': 100000,                        # train iterations
    'num_eps': 1,                               # self play times in per iter
    'explore_num': 0,                           # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.03,                    # action noise in self play games
    'update_threshold': 0.55,                   # update model threshold
    'contest_num': 10,                          # new/old model compare times
    'check_freq': 100,                          # test model frequency
    'examples_buffer_max_len': 50000,           # max length of examples buffer

    # test
    'human_color': 1                            # human player's color
}
