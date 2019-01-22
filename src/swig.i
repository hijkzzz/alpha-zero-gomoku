%module lib

%include "std_vector.i"
%template(vectori) vector<int>;
%template(vectorf) vector<double>;

%include <std_shared_ptr.i>
%shared_ptr(Gomoku);
%shared_ptr(ThreadPool);

%{
#include "gomoku.h"
#include "thread_pool.h"
#include "mcts.h"
%}

#include "gomoku.h"
#include "thread_pool.h"
#include "mcts.h"

%callback("%s_cb");
std::vector<std::vector<std::vector<double>>> infer(std::shared_ptr<Gomoku> gomoku);
%nocallback;

