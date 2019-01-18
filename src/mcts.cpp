#include <mcts.h>
#include <math.h>
#include <float.h>

#include <algorithm>

// TreeNode
TreeNode::TreeNode(std::shared_ptr<TreeNode> parent, double p_sa)
    : parent(parent), p_sa(p_sa) {}

unsigned int TreeNode::select(double c_puct) {
  double best_value = -DBL_MAX;
  unsigned int best_move = 0;

  for (auto it = this->children.begin(); it != this->children.end(); it++) {
    double cur_value = it->second->get_value(c_puct);
    if (cur_value > best_value) {
      best_value = cur_value;
      best_move = it->first;
    }
  }

  return best_move;
}

void TreeNode::expand(const std::vector<double> &action_priors) {
  for (unsigned int i = 0; i < action_priors.size(); i++) {
    // illegal action
    if (abs(action_priors[i] - 0) < DBL_EPSILON) {
      continue;
    }

    if (this->children.find(i) == this->children.end()) {
      this->children[i] = std::make_shared<TreeNode>(this->shared_from_this(),
                                                     action_priors[i]);
    }
  }
}

void TreeNode::backup(double leaf_value) {
  if (this->parent.get() != nullptr) {
    this->parent->backup(-leaf_value);
  }

  this->q_sa =
      (this->n_visited * this->q_sa + leaf_value) / (this->n_visited + 1);
  this->n_visited += 1;
}

bool TreeNode::is_leaf() { return this->children.size() == 0; }

double TreeNode::get_value(double c_puct) {
  n_visited = this->n_visited;
  double u =
      (c_puct * this->p_sa * sqrt(this->parent->n_visited) / (1 + n_visited));

  // free-lock tree search: if n_visited is 0, then ignore q_sa
  if (n_visited == 0) {
    return u + this->virtual_loss;
  }

  return this->q_sa + u + this->virtual_loss;
}

// MCTS
MCTS::MCTS(PyObject *policy_value_fn, unsigned int c_puct,
           unsigned int num_mcts_sims, std::shared_ptr<ThreadPool> thread_pool)
    : policy_value_fn(policy_value_fn), c_puct(c_puct),
      num_mcts_sims(num_mcts_sims), thread_pool(thread_pool),
      root(std::make_shared<TreeNode>(nullptr, 0)) {}

std::vector<double> MCTS::get_action_probs(std::shared_ptr<Gomoku> gomoku,
                                           double temp) {
  // submit simulate tasks to thread_pool
  std::vector<std::future<void>> futures;

  for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
    auto game = std::make_shared<Gomoku>(*gomoku);
    auto future = this->thread_pool->commit(
        std::bind(&MCTS::simulate, this, std::placeholders::_1), game);
    // future can't copy
    futures.emplace_back(std::move(future));
  }

  // wait simulate
  for (unsigned int i = 0; i < futures.size(); i++) {
    futures[i].wait();
  }

  // calc probs
  std::vector<double> action_probs(gomoku->get_action_size(), 0);
  const auto &children = this->root->children;

  // greedy
  if (abs(temp - 0) < DBL_EPSILON) {
    unsigned int max_count = 0;
    unsigned int best_action = 0;

    for (auto &pair : children) {
      if (pair.second->n_visited > max_count) {
        max_count = pair.second->n_visited;
        best_action = pair.first;
      }
    }

    action_probs[best_action] = 1.;
    return action_probs;
  }

  // explore
  double sum = 0;
  for (auto &pair : children) {
    auto action = pair.first;
    double count = pair.second->n_visited;

    if (count > 0) {
      action_probs[action] = pow(count, 1 / temp);
      sum += action_probs[action];
    }
  }

  std::for_each(action_probs.begin(), action_probs.end(),
                [sum](auto &x) { x /= sum; });

  return action_probs;
}

void MCTS::simulate(std::shared_ptr<Gomoku> game) {
  // execute one simulation

  return;
}
