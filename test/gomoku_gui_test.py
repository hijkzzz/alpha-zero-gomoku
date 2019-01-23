# -*- coding: utf-8 -*-
import sys
sys.path.append('../src')

import numpy as np
import threading
import gomoku_gui


if __name__ == "__main__":
    gomoku_gui = gomoku_gui.GomokuGUI(10)
    t = threading.Thread(target=gomoku_gui.loop)
    t.start()

    # test
    gomoku_gui.execute_move(1, (0, 0))
    gomoku_gui.execute_move(-1, (1, 0))
    gomoku_gui.execute_move(1, (3, 0))
    gomoku_gui.execute_move(-1, (4, 5))
    gomoku_gui.set_is_human(True)

    t.join()
