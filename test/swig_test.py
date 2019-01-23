# coding: utf-8

import sys
sys.path.append('../src')

import swig_wrap

class InstanceNeuralNetwork(swig_wrap.VirtualNeuralNetwork):
    # Define Python class 'constructor'
    def __init__(self):
        swig_wrap.VirtualNeuralNetwork.__init__(self)

    # Override C++ method
    def infer(self, gomoku):
        return [[[0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04], [0.5]]]


if __name__ == "__main__":
    mcts = swig_wrap.mcts()
