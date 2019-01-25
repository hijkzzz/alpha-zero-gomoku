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

    if len(sys.argv) < 2:
        print("param error: train or play")
        exit(1)

    alpha_zero = learner.Leaner(config.config)

    if sys.argv[1] == "train":
        alpha_zero.learn()
    elif sys.argv[1] == "play":
        alpha_zero.play_with_human(human_first=True)
    else:
        print("param error: train or play")
        exit(1)
