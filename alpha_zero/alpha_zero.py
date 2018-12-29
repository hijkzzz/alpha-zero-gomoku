from neural_network import NeuralNetWork
from mcts import MCTS

class AlphaZero():
    def __init__(self, game, args):
        self.game = game
        self.args = args

        self.mcts = MCTS()
        self.nnet1 = NeuralNetWork()
        self.nnet2 = NeuralNetWork()

    def learn(self):
        pass
 
