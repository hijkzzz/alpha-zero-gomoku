%module(directors="1") swig

%{
#include "gomoku.h"
#include "thread_pool.h"
#include "mcts.h"
%}

%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(DoubleVector) vector<double>;
  %template(DoubleVectorVector) vector<vector<double>>;
}

%feature("director") VirtualNeuralNetwork;

%include "gomoku.h"
%include "mcts.h"

class ThreadPool {
public:
  using task_type = std::function<void()>;

  inline ThreadPool(unsigned short thread_num = 4);
  inline ~ThreadPool();
  inline int get_idl_num();

private:
  std::vector<std::thread> pool; // thead pool
  std::queue<task_type> tasks;   // tasks queue
  std::mutex lock;               // lock for tasks queue
  std::condition_variable cv;    // condition variable for tasks queue

  std::atomic<bool> run;                    // is running
  std::atomic<unsigned int> idl_thread_num; // idle thread number
};

