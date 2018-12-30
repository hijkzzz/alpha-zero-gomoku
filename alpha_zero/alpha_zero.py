from random import shuffle
from pickle import Pickler, Unpickler
import numpy as np
from collections import deque

from neural_network import NeuralNetWork
from mcts import MCTS
from arena import Arena

class AlphaZero():
    def __init__(self, game, args):
        """args: num_mcts_sims, cpuct(mcts)
                 lr, l2, batch_size, dropout,(neural network)
                 n, nir(gomoku)
                 num_iters, num_eps, examples_max_len, threshold(self play)
        """

        self.args = args
        self.game = game
        self.nnet = NeuralNetWork(self.args)
        self.mcts = MCTS(self.game, self.nnet, self.args)

        self.train_examples = []


    def learn(self):
        for i in range(self.args.num_iters):
            train_examples = deque([], self.args.examples_max_len)


    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.

        It uses a gamma=1 if episode_step < threshold, and thereafter
        uses gamma=0.

        Returns:
            train_examples: a list of examples of the form (canonical_board,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        self.mcts = MCTS(self.game, self.nnet, self.args) # reset mcts

        train_examples = []
        board = self.game.get_init_board()
        self.cur_player = 1
        episode_step = 0

        while True:
            episode_step += 1
            canonical_board = self.game.get_canonical_form(board,self.cur_player)
            gamma = int(episode_step < self.args.threshold)

            pi = self.mcts.get_action_prob(canonical_board, gamma=gamma)
            sym = self.game.get_symmetries(canonical_board, pi)
            for b, p in sym:
                train_examples.append([b, self.cur_player, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.cur_player = self.game.get_next_state(board, self.cur_player, action)

            r = self.game.get_game_ended(board, self.cur_player)

            # END GAME
            if r != 2:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.cur_player))) for x in train_examples]