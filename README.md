# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* parallel MCTS with virtual loss
* Gomoku and MCTS are written in C++
* swig wrap python extension

## Args
```
config.py

    [gomoku]
    n = 10                                 board size
    nir = 5                                n in row

    [mcts]
    thread_pool_size = 4                   MCTS threads number
    num_mcts_sims = 400                    mcts simulation times
    c_puct = 1.5                           PUCT coeff
    c_virtual_loss = 1                     virtual loss coeff

    [neural_network]
    lr = 0.001                             learning rate
    l2 = 0.0001                            L2
    num_channels = 256                     convolution neural network channel size
    epochs = 5                             train epochs
    batch_size = 512                       batch size
    kl_targ = 0.03                         threshold of KL divergence

    [train]
    num_iters = 100000                     train iterations
    num_eps = 1                            self play times in per iter
    explore_num  = 0                       explore step in a game
    temp  = 1                              temperature
    dirichlet_alpha  = 0.03                action noise in self play games
    update_threshold = 0.55                update model threshold
    contest_num = 10                       new/old model compare times
    check_freq  = 50                       test model frequency
    examples_buffer_max_len = 10000        max length of examples buffer

    [test]
    human_color = 1                        human player's color
```

## Environment

* Python 3.6+
* PyGame 1.9+
* PyTorch 1.0+
* LibTorch 1.0+
* SWIG 3.0+
* MSVC14.00+/GCC4.8.2+
* CMake 3.0+

## Run
```
# Generate Python wrapper
cd src
swig -c++ -python .\swig.i

# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch
cmake --build

# Run
cd test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
5. github.com/suragnair/alpha-zero-general
