# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
sys.path.append('../src')

import numpy as np
import neural_network
import config


if __name__ == "__main__":
    config.args.debug = True
    config.args.action_size = config.args.n ** 2

    policy_value_net = neural_network.NeuralNetWorkWrapper(config.args)

    # test data convert

    # test loss

    # test train

    # test infer

