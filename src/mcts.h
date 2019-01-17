
#if !defined(__MCTS__)
#define __MCTS__

#include <Python.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>

#include <gomoku.h>

class TreeNode {
  public:
    TreeNode(TreeNode* parent, std::unordered_map<std::string, double> P);

    TreeNode* select();
    void expand();
    void backup();

    bool is_leaf();
    double get_value();

  private:
    TreeNode* parent;
    std::unordered_map<std::string, TreeNode*> children;

    unsigned int N;
    std::unordered_map<std::string, double> P;
    std::unordered_map<std::string, double> Q;
};

class MCTS {
  public:
    MCTS(Gomoku* gomoku, PyObject* policy_value_fn, unsigned int cpuct, unsigned int num_mcts_sims, unsigned int num_mcts_threads);
    std::vector<double> get_action_probs(double temp=1e-3);
    void search();

  private:
    TreeNode* root;
    Gomoku* gomoku;
    PyObject* policy_value_fn;

    unsigned int cpuct;
    unsigned int num_mcts_sims;
    unsigned int num_mcts_threads;
};

#endif // __MCTS__
