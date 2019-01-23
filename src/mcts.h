
#if !defined(__MCTS__)
#define __MCTS__

#include <unordered_map>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
// #include <Python.h>

#include <gomoku.h>
#include <thread_pool.h>

class TreeNode {
public:
  // friend class can access private variables
  friend class MCTS;

  TreeNode();
  TreeNode(const TreeNode &node);
  TreeNode(TreeNode *parent, double p_sa, unsigned action_size);

  TreeNode &operator=(const TreeNode &p);

  unsigned int select(double c_puct, double c_virtual_loss);
  void expand(const std::vector<double> &action_priors);
  void backup(double leaf_value);

  double get_value(double c_puct, double c_virtual_loss) const;
  inline int get_is_leaf() const { return this->is_leaf.load(); }

private:
  // store tree
  TreeNode *parent;
  std::vector<TreeNode *> children;
  std::atomic<int> is_leaf;

  // non lock
  unsigned int n_visited;
  double p_sa;
  double q_sa;

  std::atomic<int> virtual_loss;
};

// SWIG callback inferface
class VirtualNeuralNetwork {
  public:
    virtual std::vector<std::vector<std::vector<double>>> infer(Gomoku *gomoku);
    virtual ~VirtualNeuralNetwork();
};

class MCTS {
public:
  MCTS(ThreadPool* thread_pool,
       VirtualNeuralNetwork* neural_network, unsigned int c_puct,
       unsigned int num_mcts_sims, double c_virtual_loss,
       unsigned int action_size);
  std::vector<double> get_action_probs(const Gomoku* gomoku,
                                       double temp = 1e-3);
  void update_with_move(unsigned int last_move);

private:
  void simulate(std::shared_ptr<Gomoku> game);

  // variables
  TreeNode root;
  std::shared_ptr<ThreadPool> thread_pool;
  VirtualNeuralNetwork* neural_network;
  std::mutex lock;

  unsigned int action_size;
  unsigned int c_puct;
  unsigned int num_mcts_sims;
  double c_virtual_loss;
};

#endif // __MCTS__
