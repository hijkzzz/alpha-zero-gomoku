# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* free lock parallel MCTS(only atomic variable)
* Gomoku and MCTS are written in C++11
* swig wrap python extension

## Args
```
config.py

    [gomoku]
    n = 12                                 board size
    nir = 5                                n in row

    [mcts]
    thread_pool_size = 4                   C++ threads number
    num_mcts_sims = 400                    mcts simulation times
    c_puct = 5                             PUCT coeff
    c_virtual_loss = 0.1                   virtual loss coeff(see [Parallel MCTS](#References))

    [neural_network]
    lr = 0.002                             learning rate
    l2 = 0.0002                            L2
    num_channels = 128                     convolution neural network channel size
    epochs = 5                             train epochs
    batch_size = 512                       batch size
    kl_targ = 0.04                         threshold of KL divergence

    [train]
    num_iters = 100000                     train iterations
    num_eps = 1                            self play times in per iter
    explore_num  = 2                       explore step in a game
    temp  = 1                              temperature
    dirichlet_alpha  = 0.3                 action noise in self play games
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
* GCC4.8.2+/MSVC14.00+(support C++11 and compatible with Python3)
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
python learner_test.py # train model
python tester_test.py  # play with human
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## TODO
```
Because MCTS performance is limited by GIL(python) and single policy value networks, it is necessary to implement an asynchronous policy value network pool.
```


## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
5. On the Scalability of Parallel UCT
6. github.com/suragnair/alpha-zero-general
