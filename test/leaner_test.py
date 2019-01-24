# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../src')

import learner
import config

if __name__ == "__main__":

    alpha_zero = learner.Leaner(config.config)
    alpha_zero.learn()
