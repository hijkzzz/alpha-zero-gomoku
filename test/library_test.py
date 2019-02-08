# coding: utf-8
import os
import sys
sys.path.append('../build')

from library import Gomoku, MCTS

if __name__ == "__main__":
    g = Gomoku(10, 5, 1)
    g.execute_move(12)
    g.execute_move(13)
    g.execute_move(14)
    g.execute_move(15)
    g.execute_move(16)
    g.execute_move(17)
    g.execute_move(18)
    g.execute_move(22)
    g.display()

    mcts = MCTS("./models/checkpoint.pt", 4, 4, 10000, 0.5, 100, True)

    print("RUNNING")

    res = mcts.get_action_probs(g, 1)
    print(list(res))
    mcts.update_with_move(-1)
