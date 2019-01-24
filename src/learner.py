from random import shuffle, sample
import numpy as np
from collections import deque
import threading
import time

from neural_network import NeuralNetWorkWrapper
from gomoku_gui import GomokuGUI
from swig import MCTS, ThreadPool, Gomoku, VirtualNeuralNetwork

class MCTSNeuralNetwork(VirtualNeuralNetwork):
    # for swig callback inferface
    # Define Python class 'constructor'
    def __init__(self, neural_network):
        VirtualNeuralNetwork.__init__(self)
        self.neural_network = neural_network

    # Override C++ method
    def infer(self, gomoku):
        feature_batch = [gomoku.get_board()]
        res = self.neural_network.infer(feature_batch)
        return [res[0].tolist(), res[1].tolist()]

class Leaner():
    def __init__(self, args):
        # gomoku
        self.n = args.n
        self.n_in_row = args.n_in_row
        self.gomoku_gui = GomokuGUI(args.n)
        self.action_size = self.n ** 2

        # train
        self.num_iters = args.num_iters
        self.num_eps = args.num_eps
        self.check_freq = args.check_freq
        self.contest_num = args.contest_num
        self.dirichlet_alpha = args.dirichlet_alpha
        self.temp = args.temp
        self.update_threshold = args.update_threshold
        self.explore_num = args.explore_num

        self.examples_buffer = deque([], maxlen=args.examples_buffer_max_len)

        # neural network
        self.batch_size = args.batch_size

        self.nnet = NeuralNetWorkWrapper(args.lr, args.l2, args.kl_targ, args.epochs, args.num_channels, args.n, self.action_size)
        self.nnet_best = NeuralNetWorkWrapper(args.lr, args.l2, args.kl_targ, args.epochs, args.num_channels, args.n, self.action_size)

        # mcts
        self.num_mcts_sims = args.num_mcts_sims
        self.c_puct = args.c_puct
        self.c_virtual_loss = args.c_virtual_loss

        self.thread_pool = ThreadPool(args.thread_pool_size)

    def learn(self):
        self.nnet.save_model(filename="best_checkpoint")

        for i in range(1, self.num_iters + 1):
            print("ITER ::: " + str(i))

            # self play
            for eps in range(1, self.num_eps + 1):
                self.examples_buffer.extend(self.self_play())
                print("EPS :: " + str(eps) + ", EXAMPLES :: " + str(len(self.examples_buffer)))

            # sample train data
            if len(self.examples_buffer) < self.batch_size:
                continue

            print("sampling...")
            train_data = sample(self.examples_buffer, self.batch_size)

            # train neural network
            self.nnet.save_model(filename="checkpoint")
            self.nnet.train(train_data)

            if i % self.check_freq == 0:
                # compare performance
                self.nnet_best.load_model(filename="best_checkpoint")

                mcts = MCTS(self.game, self.nnet, self.args)
                mcts_old = MCTS(self.game, self.nnet_old, self.args)

                arena = Arena(mcts, mcts_old, self.game, self.board_gui)

                oneWon, twoWon, draws = arena.play_games(self.args.area_num)
                print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (oneWon, twoWon, draws))

                if oneWon + twoWon > 0 and float(oneWon) / (oneWon + twoWon) < self.args.update_threshold:
                    print('REJECTING NEW MODEL')
                else:
                    print('ACCEPTING NEW MODEL')
                    self.nnet.save_model(filename="best_checkpoint")

        # close gui
        if self.board_gui:
            self.board_gui.close_gui()
            t.join()

    def self_play(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        Returns:
            train_examples: a list of examples of the form (board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """

        train_examples = []

        board = self.game.get_init_board()
        mcts = MCTS(self.game, self.nnet, self.args)
        last_action = -1
        cur_player = self.cur_player

        episode_step = 0
        while True:
            episode_step += 1

            # temperature
            temp = self.args.temp if episode_step <= self.args.explore_num else 0
            pi, counts = mcts.get_action_prob(board, last_action, cur_player, temp=temp)

            sym = self.game.get_symmetries(board, pi)
            for b, p in sym:
                train_examples.append([b, p, last_action, cur_player])

            # Dirichlet noise
            pi_noise = 0.75 * np.array(pi)
            temp = 0.25 * np.random.dirichlet(self.args.dirichlet_alpha * np.ones(np.count_nonzero(counts)))
            j = 0
            for i, c in enumerate(counts):
                if c > 0:
                    pi_noise[i] += temp[j]
                    j += 1
            pi_noise /= np.sum(pi_noise)

            action = np.random.choice(len(pi_noise), p=pi_noise)
            last_action = action
            board, cur_player = self.game.get_next_state(board, cur_player, action)

            r = self.game.get_game_ended(board)

            # END GAME
            if r != 2:
                # b, p, v, last_action, cur_player
                return [(x[0], x[1], r * x[3], x[2], x[3]) for x in train_examples]

    def contest(self):
        pass
