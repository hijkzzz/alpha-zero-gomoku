import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        """args: num_mcts_sims, cpuct(as defined in the paper)
        """
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}       # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net)

        self.Es = {}        # cache game ended for board s
        self.Vs = {}        # cache valid moves for board s

    def get_action_prob(self, board, last_action, cur_player, temp=0):
        """
        This function performs num_mcts_sims simulations of MCTS starting from
        board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.args.num_mcts_sims):
            self.search(board, last_action, cur_player)

        s = self.game.string_representation(board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs, counts

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]
        return probs, counts

    def search(self, board, last_action, cur_player):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current board
        """

        s = self.game.string_representation(board)

        # query node type
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board)

        # TERMINAL NODE
        if self.Es[s] != 2:
            return -self.Es[s]

        # EXPAND(Not Visited)
        if s not in self.Ps:
            p_batch, v_batch = self.nnet.infer(board.reshape((1, self.args.n, self.args.n)), [last_action], [cur_player]) 
            self.Ps[s], v = p_batch[0], v_batch[0]
            valids = self.game.get_valid_moves(board, 1)
            self.Ps[s] = self.Ps[s] * valids      # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s    # renormalize

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # SELECT
        # pick the action with the highest upper confidence bound
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valids[a] == 1:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * \
                        math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    # https://applied-data.science/static/main/res/alpha_go_zero_cheat_sheet.png
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)     # default Q = 0

                if u > cur_best:
                    cur_best = u
                    best_act = a

        # DFS
        a = best_act
        next_board, next_player = self.game.get_next_state(board, cur_player, a)

        v = self.search(next_board, a, next_player)

        # BACKUP
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v)/(self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
