# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
sys.path.append('../src')

import numpy as np
import neural_network
import config


if __name__ == "__main__":
    policy_value_net = neural_network.NeuralNetWorkWrapper(config.args)

    # Neural network is not convenient for unit testing
