# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Args
```
    [gomoku]
    n = 12                                 board size
    nir = 5                                n in row

    [mcts]
    thread_pool_size = 4                    mcts threads number
    num_mcts_sims = 400                    mcts simulation times
    c_puct = 5                             PUCT coeff
    c_virtual_loss = 0.1                   virtual loss coeff

    [neural_network]
    lr = 0.002                             learning rate
    l2 = 0.0001                            L2
    epochs = 5                             train epochs
    batch_size = 512                       batch size
    num_channels = 128                     convolution neural network channel size
    kl_targ = 0.04                         threshold of KL divergence

    [train]
    num_iters = 100000                     train iterations
    num_eps = 1                            self play times in per iter
    explore_num  = 2                       explore step in a game
    temp  = 1                              temperature
    dirichlet_alpha  = 0.3                 action noise in self play games
    update_threshold = 0.55                update model threshold
    area_num = 10                          new/old model compare times
    check_freq  = 50                       test model frequency
    examples_buffer_max_len = 10000        max length of examples buffer

    [test]
    human_color = 1                        human player's color
```

## Environment
```
SWIG 3.0+
Python 3.6+
GCC4.8.2+/MSVC14.00

pytorch 0.4+
pygame
```

## Run
```
# Generate Python wrapper
cd src
swig -c++ -python .\swig.i

# Compile Python extension
python setup.py build_ext --inplace

# Run
cd ../src
TODO:

```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)


## References
* Mastering the Game of Go without Human Knowledge
* Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
* Parallel Monte-Carlo Tree Search
* A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
* On the Scalability of Parallel UCT
* github.com/suragnair/alpha-zero-general
