# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* parallel MCTS with virtual loss/libtorch
* Gomoku and MCTS are written in C++
* swig wrap python extension

## Args
Edit config.py

## Environment

* Python 3.6+
* PyGame 1.9+
* PyTorch(CUDA) 1.0+
* LibTorch(CUDA) 1.0+
* SWIG 3.0+
* MSVC14.00+/GCC4.8.2+
* CMake 3.0+

## Run
```
# Add LibTorch/SWIG to environment variable $PATH

# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DCMAKE_BUILD_TYPE=Release
cmake --build

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## Git Branchs
* master(lock parallel MCTS, libtorch)
* python(pure python, single thread)
* free-lock(free-lock parallel MCTS, requires a lot of memory)

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
5. github.com/suragnair/alpha-zero-general
