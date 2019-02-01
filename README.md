# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* free lock parallel MCTS(only one atomic variable)
* Gomoku and MCTS are written in C++
* swig wrap python extension

## Args
```
config.py

    [gomoku]
    n = 8                                 board size
    nir = 4                               n in row

    [mcts]
    thread_pool_size = 4                   C++ threads number
    num_mcts_sims = 400                    mcts simulation times
    c_puct = 3                             PUCT coeff
    c_virtual_loss = 1                     virtual loss coeff(see [Parallel MCTS](#References))

    [neural_network]
    lr = 0.001                             learning rate
    l2 = 0.0001                            L2
    num_channels = 256                     convolution neural network channel size
    epochs = 5                             train epochs
    batch_size = 512                       batch size
    kl_targ = 0.02                         threshold of KL divergence

    [train]
    num_iters = 100000                     train iterations
    num_eps = 1                            self play times in per iter
    explore_num  = 0                       explore step in a game
    temp  = 1                              temperature
    dirichlet_alpha  = 0.1                 action noise in self play games
    update_threshold = 0.55                update model threshold
    contest_num = 10                       new/old model compare times
    check_freq  = 50                       test model frequency
    examples_buffer_max_len = 10000        max length of examples buffer

    [test]
    human_color = 1                        human player's color
```

## Environment

* SWIG 3.0+
* Python 3.6+
* GCC4.8.2+ for linux/MSVC14.00+ for windows (support C++11 and compatible with Python3.6+)
* pytorch 0.4+
* pygame


## Run
```
# Generate Python wrapper
cd src
swig -c++ -python -threads .\swig.i

# Compile Python extension
python setup.py build_ext --inplace

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## TODO
* Policy Value Network with Task Queue(The performance of the parallel MCTS is limited by Python GIL and neural network

## Notice
* If your computer is out of memory, reduce the number of pre-allocated tree nodes(#define thread_object_pool_size in mcts.cpp).

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
5. github.com/suragnair/alpha-zero-general
