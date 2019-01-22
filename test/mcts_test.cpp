#include <iostream>
#include <mcts.h>

std::vector<std::vector<std::vector<double>>>
infer_test(std::shared_ptr<Gomoku> gomoku) {
  return {{{0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04}},
          {{0.5}}};
}

int main() {
  auto t = std::shared_ptr<ThreadPool>(new ThreadPool(1));
  auto g = std::shared_ptr<Gomoku>(new Gomoku(5, 3, 1));
  g->execute_move(0);
  g->execute_move(1);
  g->execute_move(2);
  g->display();

  auto m = std::shared_ptr<MCTS>(new MCTS(infer_test, 5, 1, 0.1, t));

  std::cout << "RUNNING" << std::endl;

  // auto res = m->get_action_probs(g, 1e-3);
  // std::for_each(res.begin(), res.end(), [](double x) { std::cout << x << ", "; });

  auto res = m->get_action_probs(g, 1);
  std::for_each(res.begin(), res.end(), [](double x) { std::cout << x << ", "; });

  return 0;
}
