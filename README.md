# AlphaZero Gomoku
A multi-threaded implementation of AlphaZero

## Features
* Easy Free-style Gomoku
* Tree/Root Parallelization with Virtual Loss and LibTorch
* Gomoku and MCTS are written in C++
* SWIG for Python C++ extension
* Update 2019.7.10: supporting Ubuntu and Windows

## Args
Edit config.py

## Packages

* Python 3.7
* PyGame 1.9
* PyTorch 1.2
* [LibTorch 1.2](https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.2.0.zip)
* [SWIG 3.0.12](https://sourceforge.net/projects/swig/files/)
* CMake 3.8+
* MSVC14.0/GCC6.0+

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

## Pre-trained models
> Trained 2 days on GTX1070

Link: https://pan.baidu.com/s/1c2Otxdl7VWFEXul-FyXaJA Password: e5y4

>says 啊哦，你来晚了，分享的文件已经被取消了，下次要早点哟。.


## GUI
![](https://github.com/hijkzzz/alpha-zero-gomoku/blob/master/assets/gomoku_gui.png)

## References
1. Mastering the Game of Go without Human Knowledge
2. Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
3. Parallel Monte-Carlo Tree Search
4. An Analysis of Virtual Loss in Parallel MCTS
5. A Lock-free Multithreaded Monte-Carlo Tree Search Algorithm
6. github.com/suragnair/alpha-zero-general
