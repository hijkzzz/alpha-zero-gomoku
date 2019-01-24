# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../src')

import learner
import config

if __name__ == "__main__":

    # test tuple_2d_to_numpy_2d
    # print(learner.tuple_2d_to_numpy_2d(((2, 3, 0.1, 0.5, 0), (2, 3, 0.1, 0.5, 0),
                                        # (2, 3, 0.1, 0.5, 0), (2, 3, 0.1, 0.5, 0), (2, 3, 0.1, 0.5, 0))))

    alpha_zero = learner.Leaner(config.config)
    alpha_zero.learn()
