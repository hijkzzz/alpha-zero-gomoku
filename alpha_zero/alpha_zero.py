from random import shuffle, sample
import numpy as np
from collections import deque
import threading
import time

from .neural_network import NeuralNetWorkWrapper, NeuralNetWork
from .mcts import MCTS
from .arena import Arena


class AlphaZero():
    def __init__(self, game, args, board_gui=None):
        """args: num_mcts_sims, cpuct(mcts)
                 lr, l2, batch_size, dropout, num_channels, epochs(neural network)
                 n, nir(gomoku)
                 num_iters, num_eps, temp_examples_max_len, train_examples_max_len, explore_num, area_num, update_threshold(self play)
        """

        self.args = args
        self.game = game
        self.board_gui = board_gui
        self.examples_buffer = deque([], maxlen=self.args.examples_buffer_max_len)

        self.nnet = NeuralNetWorkWrapper(NeuralNetWork(self.args), self.args)
        self.nnet.save_model(filename="best_checkpoint")
        self.nnet_old = NeuralNetWorkWrapper(NeuralNetWork(self.args), self.args)

    def learn(self):
        # start gui
        if self.board_gui:
            t = threading.Thread(target=self.board_gui.loop)
            t.start()

        for i in range(self.args.num_iters):
            print("ITER ::: " + str(i + 1))

            # self play
            self.cur_player = 1
            for eps in range(self.args.num_eps):
                self.examples_buffer.extend(self.self_play())
                self.cur_player = -self.cur_player
                print("EPS :: " + str(eps + 1) + ", EXAMPLES :: " + str(len(self.examples_buffer)))

            # sample train data
            if len(self.examples_buffer) < self.args.batch_size:
                continue

            print("sampling...")
            train_data = sample(self.examples_buffer, self.args.batch_size)

            # train neural network
            self.nnet.save_model()
            self.nnet.train(train_data)

            if (i + 1) % self.args.check_freq == 0:
                # compare performance
                self.nnet_old.load_model(filename="best_checkpoint")
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

    def human_play(self):
        t = threading.Thread(target=self.board_gui.loop)
        t.start()

        self.nnet.load_model(filename='best_checkpoint')
        mcts = MCTS(self.game, self.nnet, self.args)

        last_action = -1
        episode_step = 0

        while True:
            episode_step += 1

            # computer == player-1
            self.cur_player = -1
            pi, _ = mcts.get_action_prob(self.board_gui.board, last_action, self.cur_player)

            action = np.random.choice(len(pi), p=pi)
            board, self.cur_player = self.game.get_next_state(self.board_gui.board, self.cur_player, action)
            self.board_gui.set_board(board)

            r = self.game.get_game_ended(board)

            # END GAME
            if r != 2:
                return r

            # human == player1
            self.board_gui.human = True

            while self.board_gui.human:
                time.sleep(0.1)

            last_action = self.board_gui.last_action
            r = self.game.get_game_ended(board)

            # END GAME
            if r != 2:
                return r

        t.join()
