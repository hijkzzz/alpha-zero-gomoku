from gomoku.board_gui import BoardGUI
from gomoku.gomoku import Gomoku
from alpha_zero.alpha_zero import AlphaZero

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

args = dotdict({
    'n': 15,
    'nir': 5,

    'num_iters': 1000,
    'num_eps': 100,
    'greedy_num': 15,
    'update_threshold': 0.6,
    'area_num': 40,
    'temp_examples_max_len': 100000,
    'train_examples_max_len': 20,

    'num_mcts_sims': 25,
    'cpuct': 1,

    'lr': 0.001,
    'l2': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'num_channels': 512,
})


if __name__ == "__main__":
    game = Gomoku(args)
    args.action_size = game.get_action_size()
    board_gui = BoardGUI()

    alpha_zero = AlphaZero(game, args, board_gui)
    alpha_zero.learn()