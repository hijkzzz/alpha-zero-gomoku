from gomoku.board_gui import BoardGUI
from gomoku.gomoku import Gomoku
from alpha_zero.alpha_zero import AlphaZero

import numpy as np

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'n': 6,
    'nir': 4,

    'num_iters': 1000,
    'num_eps': 10,
    'explore_num' : 4,
    'temp' : 8,
    'dirichlet_alpha' : 0.3,
    'update_threshold': 0.55,
    'area_num': 4,
    'temp_examples_max_len': 10000,
    'train_examples_max_len': 20,

    'num_mcts_sims': 400, 
    'cpuct': 8,

    'lr': 0.001,
    'l2': 0.0001,
    'epochs': 3,
    'batch_size': 256,
    'num_channels': 128,

    'human_play' : True
})


if __name__ == "__main__":
    game = Gomoku(args)
    args['action_size'] = game.get_action_size()

    board_gui = BoardGUI(np.zeros(shape=(args.n, args.n)))

    alpha_zero = AlphaZero(game, args, board_gui)

    if not args.human_play:
        alpha_zero.learn()
    else:
        print(alpha_zero.human_play())
        input()