config = {
    # gomoku
    'n': 10,                                    # board size
    'n_in_row': 5,                              # n in row

    # mcts
    'thread_pool_size': 4,                      # MCTS threads number
    'num_mcts_sims': 800,                       # mcts simulation times
    'c_puct': 3.5,                              # PUCT coeff
    'c_virtual_loss': 0.5,                      # virtual loss coeff

    # neural_network
    'lr': 0.001,                                # learning rate
    'l2': 0.0001,                               # L2
    'num_channels': 128,                        # convolution neural network channel size
    'epochs': 5,                                # train epochs
    'batch_size': 512,                          # batch size
    'kl_targ': 0.02,                            # threshold of KL divergence

    # train
    'num_iters': 100000,                        # train iterations
    'num_eps': 1,                               # self play times in per iter
    'explore_num': 6,                           # explore step in a game
    'temp': 1,                                  # temperature
    'dirichlet_alpha': 0.03,                    # action noise in self play games
    'update_threshold': 0.55,                   # update model threshold
    'contest_num': 10,                          # new/old model compare times
    'check_freq': 100,                          # test model frequency
    'examples_buffer_max_len': 40000,           # max length of examples buffer

    # test
    'human_color': 1                            # human player's color
}
