# AlphaZero Gomoku
Gomoku AI based on AlphaZero

## Args
```
main.py

    'n': 10,                                 board size 
    'nir': 5,                                n in row

    'num_iters': 10000,                      train iterations
    'num_eps': 1,                            self play times in per iter
    'explore_num' : 2,                       explore step in a game
    'temp' : 1,                              temperature
    'dirichlet_alpha' : 0.3,                 action noise in self play games
    'update_threshold': 0.55,                update model threshold
    'area_num': 10,                          new/old model compare times
    'check_freq' : 50,                       test model frequency
    'examples_buffer_max_len': 10000,        max length of examples buffer

    'num_mcts_sims': 400,                    mcts simulation times
    'cpuct': 5,                              PUCT coeff

    'lr': 0.002,                             learning rate
    'l2': 0.0001,                            L2
    'epochs': 5,                             train epochs
    'batch_size': 512,                       batch size
    'num_channels': 128,                     convolution neural network channel size
    'kl_targ': 0.04,                         threshold of KL divergence
```

## Run
```
python main.py
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/gui.png)


## References
* Mastering the Game of Go without Human Knowledge 
* suragnair/alpha-zero-general