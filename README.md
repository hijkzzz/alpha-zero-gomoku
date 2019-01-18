# AlphaZero Gomoku
Gomoku AI based on AlphaZero(parallel MCTS)

## Args
```
config.py
    'debug': False,                          print debug information

    'n': 12,                                 board size
    'nir': 5,                                n in row

    'num_iters': 100000,                     train iterations
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
    'thread_pool_size': 4                    mcts threads number

    'lr': 0.002,                             learning rate
    'l2': 0.0001,                            L2
    'epochs': 5,                             train epochs
    'batch_size': 512,                       batch size
    'num_channels': 128,                     convolution neural network channel size
    'kl_targ': 0.04,                         threshold of KL divergence
```

## Dependencies
```
CMake 3.0+
GCC/MSVC/LLVM(C++11)

pytorch 0.4+
pygame
```

## Run
```
# Compile C++ files
mkdir build
cd build
cmake --build ..

cd ../src

# Start learning
python main.py learn

# Start testing
python main.py human
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/gui.png)


## References
* Mastering the Game of Go without Human Knowledge
* Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
* Parallel Monte-Carlo Tree Search
* A Lock-free Multithreaded Monte-Carlo TreeSearch Algorithm
* On the Scalability of Parallel UCT
* github.com/suragnair/alpha-zero-general
