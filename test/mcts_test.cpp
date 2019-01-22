#include <iostream>
#include <mcts.h>

std::vector<std::vector<std::vector<double>>>
infer_test(std::shared_ptr<Gomoku> gomoku) {
  return {{{0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
            0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04}},
          {{0.5}}};
}

int main() {
  auto t = std::shared_ptr<ThreadPool>(new ThreadPool(4));
  auto g = std::shared_ptr<Gomoku>(new Gomoku(5, 3, 1));
  g->execute_move(0);
  g->execute_move(1);
  g->execute_move(2);
  g->display();

  auto m = std::shared_ptr<MCTS>(new MCTS(t, infer_test, 5, 10000, 0.1, 5 * 5));

  std::cout << "RUNNING" << std::endl;

  // auto res = m->get_action_probs(g, 1e-3);
  // std::for_each(res.begin(), res.end(), [](double x) { std::cout << x << ", "; });

  auto res = m->get_action_probs(g, 1);
  std::for_each(res.begin(), res.end(), [](double x) { std::cout << x << ", "; });

  return 0;
}
