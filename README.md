# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* Easy Free-style Gomoku
* Multi-threading Tree/Root Parallelization with Virtual Loss and LibTorch
* Gomoku, MCTS and Network Infer are written in C++
* SWIG for Python C++ extension
* Update 2019.7.10: Supporting Ubuntu and Windows
* Update 2022.4.4: Re-compile with CUDA 11.6/PyTorch 1.10/LibTorch 1.10(Pre-cxx11 ABI)/SWIG 4.0.2

## Args
Edit config.py

## Packages
* Python 3.6+
* PyGame 1.9+
* CUDA 10+
* [PyTorch 1.1+](https://pytorch.org/get-started/locally/)
* [LibTorch 1.1+ (Pre-cxx11 ABI)](https://pytorch.org/get-started/locally/)
* [SWIG 3.0.12+](https://sourceforge.net/projects/swig/files/)
* CMake 3.8+
* MSVC14.0+ / GCC6.0+

## Run
```
# Compile Python extension
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=path/to/libtorch -DPYTHON_EXECUTABLE=path/to/python -DCMAKE_BUILD_TYPE=Release
make -j10

# Run
cd ../test
python learner_test.py train # train model
python learner_test.py play  # play with human
```

## Pre-trained models
> Trained 2 days on GTX TITAN X (similar to GTX1070)

See GitHub Release: https://github.com/hijkzzz/alpha-zero-gomoku/releases


## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
6. github.com/suragnair/alpha-zero-general
