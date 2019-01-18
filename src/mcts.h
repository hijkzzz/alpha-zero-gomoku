
#if !defined(__MCTS__)
#define __MCTS__

#include <Python.h>
#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>

#include <gomoku.h>
#include <thread_pool.h>

class TreeNode : std::enable_shared_from_this<TreeNode> {
  public:
    TreeNode(std::shared_ptr<TreeNode> parent, double p_sa);

    unsigned int select(double c_puct);
    void expand(const std::vector<double>& action_priors);
    void backup(double leaf_value);

    bool is_leaf();
    double get_value(double c_puct);

  private:
    // store tree
    std::shared_ptr<TreeNode> parent;
    std::unordered_map<unsigned int, std::shared_ptr<TreeNode>> children;

    // non lock
    unsigned int n_visited;
    double p_sa;
    double q_sa;

    std::atomic<double> virtual_loss;
};

class MCTS {
  public:
    MCTS(PyObject* policy_value_fn, unsigned int c_puct, unsigned int num_mcts_sims, std::shared_ptr<ThreadPool> thread_pool);
    std::vector<double> get_action_probs(std::shared_ptr<Gomoku> gomoku, double temp=1e-3);
    void search(std::vector<std::vector<int>>& board);

  private:
    std::shared_ptr<TreeNode> root;
    std::shared_ptr<ThreadPool> thread_pool;
    PyObject* policy_value_fn;

    unsigned int c_puct;
    unsigned int num_mcts_sims;
    unsigned int num_mcts_threads;
};

#endif // __MCTS__
