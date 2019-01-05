from random import shuffle
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
        self.train_examples = []

        self.nnet = NeuralNetWorkWrapper(NeuralNetWork(self.args), self.args)
        self.nnet_old = NeuralNetWorkWrapper(NeuralNetWork(self.args), self.args)


    def learn(self):
        # start gui
        if self.board_gui:
            t = threading.Thread(target=self.board_gui.loop)
            t.start()

        for i in range(self.args.num_iters):
            print("ITER :::: " + str(i + 1))

            # self play
            temp_examples = deque([], maxlen=self.args.temp_examples_max_len)

            for eps in range(self.args.num_eps):
                temp_examples += self.self_play()
                print("EPS ::: " + str(eps + 1) + ", EXAMPLES ::: " + str(len(temp_examples)))
            
            # add to train examples
            self.train_examples.append(temp_examples)

            if len(self.train_examples) > self.args.train_examples_max_len:
                self.train_examples.pop(0)

            # sample train data
            train_data = []
            for e in self.train_examples:
                train_data.extend(e)

            shuffle(train_data)

            # train neural network
            self.nnet.save_model()
            self.nnet_old.load_model()

            self.nnet.train(train_data)

            # compare performance
            mcts = MCTS(self.game, self.nnet, self.args)
            mcts_old = MCTS(self.game, self.nnet_old, self.args)
            
            arena = Arena(lambda x: np.argmax(mcts.get_action_prob(x)[0]),
                          lambda x: np.argmax(mcts_old.get_action_prob(x)[0]), 
                          self.game,
                          self.board_gui)

            oneWon, twoWon, draws = arena.play_games(self.args.area_num)
            print("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (oneWon, twoWon, draws))

            if oneWon + twoWon > 0 and float(oneWon) / (oneWon + twoWon) < self.args.update_threshold:
                print('REJECTING NEW MODEL')
                self.nnet.load_model()
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
            train_examples: a list of examples of the form (canonical_board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """

        train_examples = []

        board = self.game.get_init_board()
        mcts = MCTS(self.game, self.nnet, self.args)
        self.cur_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board, self.cur_player)

            # temperature
            temp = 1 if episode_step < self.args.explore_num else 0
            temp *= self.args.temp
            pi, counts = mcts.get_action_prob(canonical_board, temp=temp)

            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, self.cur_player, p, None])

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
            board, self.cur_player = self.game.get_next_state(board, self.cur_player, action)

            r = self.game.get_game_ended(board, self.cur_player)

            # END GAME
            if r != 2:
                # b, p, v
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.cur_player))) for x in train_examples]


    def human_play(self):
        t = threading.Thread(target=self.board_gui.loop)
        t.start()

        self.nnet.load_model(filename='best_checkpoint')
        mcts = MCTS(self.game, self.nnet, self.args)

        self.cur_player = -1
        episode_step = 0

        while True:
            episode_step += 1

            # computer == player-1
            self.cur_player = -1
            canonical_board = self.game.get_canonical_form(self.board_gui.board, self.cur_player)
            pi, _ = mcts.get_action_prob(canonical_board)

            action = np.random.choice(len(pi), p=pi)
            board, self.cur_player = self.game.get_next_state(self.board_gui.board, self.cur_player, action)
            self.board_gui.set_board(board)

            r = self.game.get_game_ended(board, self.cur_player)

            # END GAME
            if r != 2:
                return r

            # human == player1
            self.board_gui.human = True

            while self.board_gui.human:
                time.sleep(0.1)

            r = self.game.get_game_ended(board, self.cur_player)

            # END GAME
            if r != 2:
                return r
        
        t.join()