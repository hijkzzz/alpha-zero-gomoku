# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../src')
import configparser

import learner

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    alpha_zero = learner.Leaner(config)
    alpha_zero.learn()
