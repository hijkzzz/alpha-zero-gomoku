
#if !defined(__MCTS__)
#define __MCTS__

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <memory>
#include <Python.h>

#include <gomoku.h>
#include <thread_pool.h>

class TreeNode : std::enable_shared_from_this<TreeNode> {
public:
  // friend class can access private variables
  friend class MCTS;

  TreeNode(std::shared_ptr<TreeNode> parent, double p_sa);

  unsigned int select(double c_puct, double c_virual_loss);
  void expand(const std::vector<double> &action_priors);
  void backup(double leaf_value);

  bool is_leaf() const;
  double get_value(double c_puct, double c_virual_loss) const;

private:
  // store tree
  std::shared_ptr<TreeNode> parent;
  std::unordered_map<unsigned int, std::shared_ptr<TreeNode>> children;

  // non lock
  unsigned int n_visited;
  double p_sa;
  double q_sa;

  std::atomic<int> virtual_loss;
};

class MCTS {
public:
  using function_type = std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> (*)(std::shared_ptr<Gomoku> gomoku);

  MCTS(function_type neural_network_infer, unsigned int c_puct,
       unsigned int num_mcts_sims, double c_virtual_loss,
       std::shared_ptr<ThreadPool> thread_pool);
  std::vector<double> get_action_probs(std::shared_ptr<const Gomoku> gomoku,
                                       double temp = 1e-3);
  void simulate(std::shared_ptr<Gomoku> game);
  void reset(unsigned int last_move);
  std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> infer(std::shared_ptr<Gomoku> game);

private:
  std::shared_ptr<TreeNode> root;
  std::shared_ptr<ThreadPool> thread_pool; // compatible with C
  function_type neural_network_infer;

  unsigned int c_puct;
  unsigned int num_mcts_sims;
  double c_virtual_loss;
};

#endif // __MCTS__
