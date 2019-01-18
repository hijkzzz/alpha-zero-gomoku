#include <mcts.h>
#include <math.h>
#include <float.h>

// TreeNode
TreeNode::TreeNode(std::shared_ptr<TreeNode> parent, double p_sa)
    : parent(parent), p_sa(p_sa){}

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

void TreeNode::expand(const std::vector<double>& action_priors) {
  for(unsigned int i = 0; i < action_priors.size(); i++) {
    // illegal action
    if (abs(action_priors[i] - 0) < DBL_EPSILON ) {
      continue;
    }

    if (this->children.find(i) == this->children.end()) {
      this->children[i] = std::make_shared<TreeNode>(this->shared_from_this(), action_priors[i]);
    }
  }
}

void TreeNode::backup(double leaf_value) {
  if (this->parent.get() != nullptr) {
    this->parent->backup(-leaf_value);
  }

  this->q_sa = (this->n_visited * this->q_sa + leaf_value) / (this->n_visited + 1);
  this->n_visited += 1;
}

bool TreeNode::is_leaf() { return this->children.size() == 0; }

double TreeNode::get_value(double c_puct) {
  double u = (c_puct * this->p_sa * sqrt(this->parent->n_visited) /
              (1 + this->n_visited));

  return this->q_sa + u + this->virtual_loss;
}


// MCTS
MCTS::MCTS(PyObject* policy_value_fn, unsigned int c_puct, unsigned int num_mcts_sims,  std::shared_ptr<ThreadPool> thread_pool) {

}

std::vector<double> MCTS::get_action_probs(std::shared_ptr<Gomoku> gomoku, double temp=1e-3) {

}

void MCTS::search(std::vector<std::vector<int>>& board) {

}
