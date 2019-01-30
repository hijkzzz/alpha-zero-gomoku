#include <mcts.h>
#include <math.h>
#include <float.h>

#include <numeric>
#include <thread>
#include <iostream>

// thread local object pool
#define thread_object_pool_size 1000000
thread_local std::vector<TreeNode> thread_object_pool(thread_object_pool_size);
thread_local unsigned int thread_object_pool_index = 0;

// TreeNode
TreeNode::TreeNode()
    : is_leaf(1), virtual_loss(0), q_sa(0), p_sa(0), n_visited(0) {}

TreeNode::TreeNode(
    const TreeNode &node) {  // because automic<>, define copy function
  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited = node.n_visited;
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;

  this->virtual_loss.store(node.virtual_loss.load());
}

TreeNode::TreeNode(TreeNode *parent, double p_sa, unsigned int action_size)
    : parent(parent),
      p_sa(p_sa),
      children(action_size, nullptr),
      is_leaf(true),
      virtual_loss(0),
      q_sa(0),
      n_visited(0) {}

TreeNode &TreeNode::operator=(const TreeNode &node) {
  if (this == &node) {
    return *this;
  }

  // struct
  this->parent = node.parent;
  this->children = node.children;
  this->is_leaf = node.is_leaf;

  this->n_visited = node.n_visited;
  this->p_sa = node.p_sa;
  this->q_sa = node.q_sa;
  this->virtual_loss.store(node.virtual_loss.load());

  return *this;
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss) {
  double best_value = -DBL_MAX;
  unsigned int best_move = 0;
  TreeNode *best_node;

  for (unsigned int i = 0; i < this->children.size(); i++) {
    // empty node
    if (children[i] == nullptr) {
      continue;
    }

    double cur_value = children[i]->get_value(c_puct, c_virtual_loss);
    if (cur_value > best_value) {
      best_value = cur_value;
      best_move = i;
      best_node = children[i];
    }
  }

  // add vitural loss
  best_node->virtual_loss--;

  return best_move;
}

void TreeNode::expand(const std::vector<double> &action_priors) {
  auto action_size = this->children.size();

  for (unsigned int i = 0; i < action_size; i++) {
    // illegal action
    if (abs(action_priors[i] - 0) < FLT_EPSILON) {
      continue;
    }

    if (this->children[i] == nullptr) {
      // get object from object pool
      TreeNode *new_node = &thread_object_pool[thread_object_pool_index];
      thread_object_pool_index =
          (thread_object_pool_index + 1) % thread_object_pool_size;

      new_node->parent = this;
      new_node->p_sa = action_priors[i];
      new_node->children = std::vector<TreeNode *>(action_size, nullptr);
      new_node->is_leaf = true;

      this->children[i] = new_node;
    }
  }

  // expand, not leaf
  this->is_leaf = false;
}

void TreeNode::backup(double value) {
  // If it is not root, this node's parent should be updated first
  if (this->parent != nullptr) {
    this->parent->backup(-value);
  }

  this->q_sa = (this->n_visited * this->q_sa + value) / (this->n_visited + 1);
  this->n_visited += 1;
}

double TreeNode::get_value(double c_puct, double c_virtual_loss) const {
  auto n_visited = this->n_visited;

  unsigned int sum_n_visited = 0;
  std::for_each(this->parent->children.begin(), this->parent->children.end(),
                [&sum_n_visited](TreeNode *node) {
                  sum_n_visited += node ? node->n_visited : 0;
                });

  double u = (c_puct * this->p_sa * sqrt(sum_n_visited) / (1 + n_visited));

  // free-lock tree search: if n_visited is 0, then ignore q_sa
  if (n_visited == 0) {
    return u;
  } else {
    double virtual_loss =
        c_virtual_loss * this->virtual_loss.load() / n_visited;
    return this->q_sa + u + virtual_loss;
  }
}

// MCTS
MCTS::MCTS(ThreadPool *thread_pool, VirtualNeuralNetwork *neural_network,
           unsigned int c_puct, unsigned int num_mcts_sims,
           double c_virtual_loss, unsigned int action_size)
    : neural_network(neural_network),
      c_puct(c_puct),
      num_mcts_sims(num_mcts_sims),
      c_virtual_loss(c_virtual_loss),
      thread_pool(thread_pool),
      action_size(action_size),
      root(nullptr, 1., action_size) {}

void MCTS::update_with_move(int last_action) {
  // reset the tree
  auto &children = this->root.children;

  // reuse the child tree
  if (last_action >= 0 && children[last_action] != nullptr) {
    this->root = *children[last_action];
    this->root.parent = nullptr;

  } else {
    this->root = TreeNode(nullptr, 1., this->action_size);
  }
}

std::vector<double> MCTS::get_action_probs(Gomoku *gomoku, double temp) {
  // submit simulate tasks to thread_pool
  std::vector<std::future<void>> futures;

  for (unsigned int i = 0; i < this->num_mcts_sims; i++) {
    // copy gomoku
    auto game = std::make_shared<Gomoku>(*gomoku);
    auto future =
        this->thread_pool->commit(std::bind(&MCTS::simulate, this, game));

    // future can't copy
    futures.emplace_back(std::move(future));
  }

  // wait simulate
  for (unsigned int i = 0; i < futures.size(); i++) {
    futures[i].wait();
  }

  // calculate probs
  std::vector<double> action_probs(gomoku->get_action_size(), 0);
  const auto &children = this->root.children;

  // greedy
  if (temp - 1e-3 < FLT_EPSILON) {
    unsigned int max_count = 0;
    unsigned int best_action = 0;

    for (unsigned int i = 0; i < children.size(); i++) {
      if (children[i] && children[i]->n_visited > max_count) {
        max_count = children[i]->n_visited;
        best_action = i;
      }
    }

    action_probs[best_action] = 1.;
    return action_probs;

  } else {
    // explore
    double sum = 0;
    for (unsigned int i = 0; i < children.size(); i++) {
      if (children[i] && children[i]->n_visited > 0) {
        action_probs[i] = pow(children[i]->n_visited, 1 / temp);
        sum += action_probs[i];
      }
    }

    // renormalization
    std::for_each(action_probs.begin(), action_probs.end(),
                  [sum](double &x) { x /= sum; });

    return action_probs;
  }
}

void MCTS::simulate(std::shared_ptr<Gomoku> game) {
  // execute one simulation
  auto node = &this->root;

  while (true) {
    if (node->get_is_leaf()) {
      break;
    }

    // select
    auto action = node->select(this->c_puct, this->c_virtual_loss);
    game->execute_move(action);
    node = node->children[action];
  }

  // get game status
  auto status = game->get_game_status();
  double value = 0;

  // not end
  if (!status[0]) {
    // predict action_probs and value by neural network
    std::vector<double> action_priors(this->action_size, 0);

    {
      // std::lock_guard<std::mutex> lock(this->lock);
      auto res = std::move(this->neural_network->infer(game.get()));

      action_priors = std::move(res[0]);
      value = res[1][0];
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

    // renormalization
    if (sum > FLT_EPSILON) {
      std::for_each(action_priors.begin(), action_priors.end(),
                    [sum](double &x) { x /= sum; });
    } else {
      // all masked

      // NB! All valid moves may be masked if either your NNet architecture is
      // insufficient or you've get overfitting or something else. If you have
      // got dozens or hundreds of these messages you should pay attention to
      // your NNet and/or training process.
      std::cout << "All valid moves were masked, do workaround." << std::endl;

      sum = std::accumulate(legal_moves.begin(), legal_moves.end(), 0);
      for (unsigned int i = 0; i < action_priors.size(); i++) {
        action_priors[i] = legal_moves[i] / sum;
      }
    }

    // expand
    node->expand(action_priors);

  } else {
    // end
    auto winner = status[1];
    value = (winner == 0 ? 0 : (winner == game->get_current_color() ? 1 : -1));
  }

  // backup, -value because game->get_current_color() is next player
  node->backup(-value);
  return;
}
