# AlphaZero Gomoku
Gomoku AI based on AlphaZero

## Args
```
main.py

    'n': 7,                                  board size 
    'nir': 3,                                n in row

    'num_iters': 1000,                       train iterations
    'num_eps': 150,                          self play times in per iter
    'explore_num' : 5,                       explore step in a game
    'dirichlet_alpha' : 0.3,                 action noise in self play games
    'update_threshold': 0.55,                update model threshold
    'area_num': 20,                          new/old model compare times
    'temp_examples_max_len': 100000,         max length of examples in a game
    'train_examples_max_len': 5,             max length of history games' records

    'num_mcts_sims': 150,                    mcts simulation times
    'cpuct': 5,                              PUCT coeff

    'lr': 0.001,                             learning rate
    'l2': 0.0001,                            L2
    'epochs': 3,                             train epochs
    'batch_size': 64,                        batch size
    'num_channels': 128,                     convolution neural network channel size

    'human_play' : False                     play with human(test mode)
```

## Run
```
python main.py
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/gui.png)