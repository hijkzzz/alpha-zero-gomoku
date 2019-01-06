# AlphaZero Gomoku
Gomoku AI based on AlphaZero

## Args
```
main.py

    'n': 6,                                  board size 
    'nir': 4,                                n in row

    'num_iters': 3000,                       train iterations
    'num_eps': 1,                            self play times in per iter
    'explore_num' : 2,                       explore step in a game
    'temp' : 10,                             temperature
    'dirichlet_alpha' : 0.3,                 action noise in self play games
    'update_threshold': 0.55,                update model threshold
    'area_num': 10,                          new/old model compare times
    'check_freq' : 50,                       test model frequency
    'examples_buffer_max_len': 10000,        max length of examples in buffer

    'num_mcts_sims': 400,                    mcts simulation times
    'cpuct': 10,                             PUCT coeff

    'lr': 0.003,                             learning rate
    'l2': 0.0001,                            L2
    'epochs': 5,                             train epochs
    'batch_size': 512,                       batch size
    'num_channels': 128,                     convolution neural network channel size
    'kl_targ': 0.02,                         threshold of KL(old model, new model)

    'human_play' : False                     play with human(GUI)
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