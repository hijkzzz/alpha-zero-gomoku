# AlphaZero Gomoku
Gomoku AI based on AlphaZero

## Args
```
main.py

    'n': 6,                                  board size 
    'nir': 4,                                n in row

    'num_iters': 1000,                       train iterations
    'num_eps': 10,                           self play times in per iter
    'explore_num' : 4,                       explore step in a game
    'temp' : 10,                             temperature
    'dirichlet_alpha' : 0.3,                 action noise in self play games
    'update_threshold': 0.55,                update model threshold
    'area_num': 4,                           new/old model compare times
    'temp_examples_max_len': 10000,          max length of examples in a game
    'train_examples_max_len': 20,            max length of history games' records

    'num_mcts_sims': 400,                    mcts simulation times
    'cpuct': 10,                             PUCT coeff

    'lr': 0.003,                             learning rate
    'l2': 0.0001,                            L2
    'epochs': 5,                             train epochs
    'batch_size': 512,                       batch size
    'num_channels': 128,                     convolution neural network channel size

    'human_play' : False                     play with human(test mode)
```

## Run
```
python main.py
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/gui.png)

## TODO
* multiprocess(asynchronous pipeline)
* numba

## References
* Mastering the Game of Go without Human Knowledge 
* suragnair/alpha-zero-general