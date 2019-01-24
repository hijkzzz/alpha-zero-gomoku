# coding: utf-8

import sys
sys.path.append('../src')
import numpy as np

import swig

class TestNeuralNetwork(swig.VirtualNeuralNetwork):
    # Define Python class 'constructor'
    def __init__(self):
        swig.VirtualNeuralNetwork.__init__(self)

    # Override C++ method
    def infer(self, gomoku):
        return [[0.04 for _ in range(100)], [0.]]


if __name__ == "__main__":
    t = swig.ThreadPool(4)
    print(t.get_idl_num())

    g = swig.Gomoku(10, 5, -1)
    g.execute_move(49)
    g.execute_move(50)
    g.display()

    # print(g.get_board())
    # print(g.get_current_color())

    n = TestNeuralNetwork()
    m = swig.MCTS(t, n, 5, 800, 0.1, g.get_action_size())

    while g.get_game_status()[0] == 0:
        res = m.get_action_probs(g, 0)
        # res = m.get_action_probs(g, 1)
        best_move =  int(np.argmax(np.array(list(res))))

        print(g.get_last_move())
        # print(best_move)
        g.execute_move(best_move)
        m.update_with_move(best_move)

    g.display()
