# coding: utf-8

import sys
sys.path.append('../src')

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
    m = swig.MCTS(t, n, 5, 100, 0.1, g.get_action_size())

    print(g.get_game_status())

    res = m.get_action_probs(g, 1)
    print(res)
