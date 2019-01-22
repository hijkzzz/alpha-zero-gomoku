#include <mcts.h>
#include <math.h>
#include <float.h>

#include <numeric>
#include <iostream>

// TreeNode
TreeNode::TreeNode(TreeNode* parent, double p_sa)
    : parent(parent), p_sa(p_sa) {}

TreeNode::~TreeNode() {
  // clean sub tree
  for(auto &pair : this->children){
    delete pair.second;
  }
}

unsigned int TreeNode::select(double c_puct, double c_virtual_loss) {
  double best_value = -DBL_MAX;
  unsigned int best_move = 0;
  TreeNode* best_node;

  for (auto it = this->children.begin(); it != this->children.end(); it++) {
    double cur_value = it->second->get_value(c_puct, c_virtual_loss);
    if (cur_value > best_value) {
      best_value = cur_value;
      best_move = it->first;
      best_node = it->second;
    }
  }

  // add vitural loss
  best_node->virtual_loss--;

  return best_move;
}

void TreeNode::expand(const std::vector<double> &action_priors) {
  // std::for_each(action_priors.begin(), action_priors.end(), [](double x) {
  // std::cout << x << ", "; }); std::cout << std::endl;

  for (unsigned int i = 0; i < action_priors.size(); i++) {
    // illegal action
    if (abs(action_priors[i] - 0) < DBL_EPSILON) {
      continue;
    }

    this->children[i] = new TreeNode(this, action_priors[i]);
  }
}

void TreeNode::backup(double value) {
  this->q_sa = (this->n_visited * this->q_sa + value) / (this->n_visited + 1);
  this->n_visited += 1;

  if (this->parent != nullptr) {
    // remove virtual loss
    this->virtual_loss++;

    this->parent->backup(-value);
  }
}

bool TreeNode::is_leaf() const { return this->children.size() == 0; }

double TreeNode::get_value(double c_puct, double c_virtual_loss) const {
  auto n_visited = this->n_visited;
  double u =
      (c_puct * this->p_sa * sqrt(this->parent->n_visited) / (1 + n_visited));

  auto virtual_loss = this->virtual_loss.load() * c_virtual_loss;

  // free-lock tree search: if n_visited is 0, then ignore q_sa
  if (n_visited == 0) {
    return u + virtual_loss;
  }

  return this->q_sa + u + virtual_loss;
}

// MCTS
MCTS::MCTS(function_type neural_network_infer, unsigned int c_puct,
           unsigned int num_mcts_sims, double c_virtual_loss,
           std::shared_ptr<ThreadPool> thread_pool)
    : neural_network_infer(neural_network_infer), c_puct(c_puct),
      num_mcts_sims(num_mcts_sims), c_virtual_loss(c_virtual_loss),
      thread_pool(thread_pool), root(new TreeNode(nullptr, 1.)) {}

std::vector<double> MCTS::get_action_probs(std::shared_ptr<const Gomoku> gomoku,
                                           double temp) {
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
  const auto &children = this->root->children;

  // greedy
  if (temp - 1e-3 < DBL_EPSILON) {
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

  // renormalization
  std::for_each(action_probs.begin(), action_probs.end(),
                [sum](double &x) { x /= sum; });

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
    auto action = node->select(this->c_puct, this->c_virtual_loss);
    game->execute_move(action);
  }

  // get game status
  auto status = game->get_game_status();
  double value = 0;

  // not end
  if (!status[0]) {
    // predict action_probs and value by neural network
    auto res = std::move(this->neural_network_infer(game));
    std::vector<double> action_priors = std::move(res[0][0]);
    value = res[1][0][0];

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
    if (sum > DBL_EPSILON) {
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

MCTS::~MCTS() {
  delete this->root;
}

void MCTS::reset(unsigned int last_action) {
  // reset the tree
  auto &children = this->root->children;
  TreeNode* new_root;

  // reuse the child tree
  if (children.find(last_action) != children.end()) {
    new_root = children[last_action];
    // unlink parent
    children.erase(last_action);
    new_root->parent = nullptr;
  } else {
    new_root = new TreeNode(nullptr, 1.);
  }

  // release old tree
  delete this->root;
  this->root = new_root;
}
