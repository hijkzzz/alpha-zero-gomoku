%module(directors="1") swig_lib

%{
#define SWIG_PYTHON_THREADS

#include "gomoku.h"
#include "thread_pool.h"
#include "mcts.h"
%}

%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(DoubleVector) vector<double>;
  %template(DoubleDoubleVector) vector<vector<double>>;
  %template(DoubleVectorVectorVector) vector<vector<vector<double>>>;
}

%feature("director") VirtualNeuralNetwork;

#include "gomoku.h"
#include "thread_pool.h"
#include "mcts.h"
