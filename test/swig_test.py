# coding: utf-8

import sys
sys.path.append('../src')
import numpy as np

import swig

class InstanceNeuralNetwork(swig.VirtualNeuralNetwork):
    # Define Python class 'constructor'
    def __init__(self):
        swig.VirtualNeuralNetwork.__init__(self)

    # Override C++ method
    def infer(self, gomoku):
        return [[[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]], [[0.5]]]


if __name__ == "__main__":
    t = swig.ThreadPool(1)
    print(t.get_idl_num())

    g = swig.Gomoku(5, 3, 1)
    g.execute_move(12)
    g.execute_move(13)
    g.display()

    n = InstanceNeuralNetwork()
    m = swig.MCTS(t, n, 5, 10000, 0.1, g.get_action_size())

    while g.get_game_status()[0] == 0:
        res = m.get_action_probs(g, 1)
        best_move =  int(np.argmax(np.array(list(res))))

        print(best_move)
        g.execute_move(best_move)
        m.update_with_move(best_move)
        g.display()
