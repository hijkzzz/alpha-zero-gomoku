class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

args = dotdict({
    'debug': True,

    'n': 10,
    'nir': 5,

    'num_iters': 100000,
    'num_eps': 1,
    'explore_num' : 2,
    'temp' : 1,
    'dirichlet_alpha' : 0.3,
    'update_threshold': 0.55,
    'area_num': 10,
    'check_freq' : 50,
    'examples_buffer_max_len': 10000,

    'num_mcts_sims': 400,
    'num_mcts_threads': 4,
    'cpuct': 5,

    'lr': 0.002,
    'l2': 0.0002,
    'epochs': 5,
    'batch_size': 512,
    'num_channels': 128,
    'kl_targ': 0.04,

    'human_color': 1
})


args.action_size = args.n ** 2
