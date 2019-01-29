config = {
    # gomoku
    'n': 8,
    'n_in_row': 5,

    # mcts
    'thread_pool_size': 4,
    'num_mcts_sims': 400,
    'c_puct': 5,
    'c_virtual_loss': 0.3,

    # neural_network
    'lr': 0.001,
    'l2': 0.0001,
    'num_channels': 128,
    'epochs': 5,
    'batch_size': 512,
    'kl_targ': 0.02,

    # train
    'num_iters': 100000,
    'num_eps': 1,
    'explore_num': 2,
    'temp': 1,
    'dirichlet_alpha': 0.3,
    'update_threshold': 0.55,
    'contest_num': 10,
    'check_freq': 50,
    'examples_buffer_max_len': 20000,

    # test
    'human_color': 1
}
