#include <iostream>
#include <mcts.h>

int main() {
  auto g = std::make_shared<Gomoku>(10, 5, 1);
  g->execute_move(12);
  g->execute_move(13);
  g->execute_move(14);
  g->execute_move(15);
  g->execute_move(16);
  g->execute_move(17);
  g->execute_move(18);
  g->execute_move(19);
  g->display();

  auto m = std::make_shared<MCTS>("../test/models/checkpoint.pt", 4, 1.5, 400, 1,
                                  g->get_action_size());

  std::cout << "RUNNING" << std::endl;

  while (true) {
    auto res = m->get_action_probs(g.get(), 1);
    std::for_each(res.begin(), res.end(),
                  [](double x) { std::cout << x << ","; });
    std::cout << std::endl;
    m->update_with_move(-1);
  }

  return 0;
}
