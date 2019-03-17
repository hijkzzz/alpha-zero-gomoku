# coding: utf-8
import sys
sys.path.append('..')
sys.path.append('../src')

import learner
import config

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ["train", "play"]:
        print("[USAGE] python leaner_test.py train|play")
        exit(1)

    alpha_zero = learner.Leaner(config.config)

    if sys.argv[1] == "train":
        alpha_zero.learn()
    elif sys.argv[1] == "play":
        for i in range(10):
            print("GAME: {}".format(i + 1))
            alpha_zero.play_with_human(human_first=i % 2)
