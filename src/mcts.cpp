#include <mcts.h>
#include <math.h>
#include <float.h>

#include <numeric>
#include <iostream>

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
      root(std::make_shared<TreeNode>(nullptr, 1.)) {}

std::vector<double> MCTS::get_action_probs(std::shared_ptr<Gomoku> gomoku,
                                           double temp) {
  // submit simulate tasks to thread_pool
  std::vector<std::future<void>> futures;

  for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
    // copy gomoku
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
  auto node = this->root;

  while (true) {
    if (node->is_leaf()) {
      break;
    }

    // select
    auto action = node->select(this->c_puct);
    auto n = game->get_n();
    game->execute_move(std::make_tuple(action / n, action % n));
  }

  // predict action_probs and value by neural network
  std::vector<double> action_priors;
  double value;

  {
    // TODO: call python
  }

  // mask invalid actions
  auto legal_moves = game->get_legal_moves();
  double sum = 0;
  for (unsigned int i = 0; i < action_priors.size(); i++) {
    if (legal_moves[i] == 1) {
      sum += action_priors[i];
    } else {
      action_priors[i] = 0;
    }
  }

  if (sum > DBL_EPSILON) {
    std::for_each(action_priors.begin(), action_priors.end(),
                  [sum](auto &x) { x /= sum; });
  } else {
    // all masked

    // NB! All valid moves may be masked if either your NNet architecture is
    // insufficient or you've get overfitting or something else. If you have got
    // dozens or hundreds of these messages you should pay attention to your
    // NNet and/or training process.
    std::cout << "All valid moves were masked, do workaround." << std::endl;

    sum = std::accumulate(legal_moves.begin(), legal_moves.end());
    for (unsigned int i = 0; i < action_priors.size(); i++) {
      action_priors[i] = legal_moves[i] / sum;
    }
  }

  // get game status
  auto status = game->get_game_status();

  // not end
  if (!std::get<0>(status)) {
    // expand
    node->expand(action_priors);
  } else {
    // end
    auto winner = std::get<1>(status);
    value = (winner == 0 ? 0 : (winner == game->get_current_color() ? 1 : -1));
  }

  // backup, -value because game->get_current_color() is next player
  node->backup(-value);
  return;
}

void MCTS::reset(unsigned int last_action) {
  // reset the tree

  auto &children = this->root->children;
  // reuse the child tree
  if (children.find(last_action) != children.end()) {
    this->root = children[last_action];
  } else {
    this->root = std::make_shared<TreeNode>(nullptr, 1.);
  }
}
