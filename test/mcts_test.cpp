#include <iostream>
#include <mcts.h>

class InstanceNeuralNetwork : public VirtualNeuralNetwork {
public:
  std::vector<std::vector<std::vector<double>>> infer(Gomoku *gomoku) {
    return {{{0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04,
              0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04}},
            {{0.5}}};
  }
};

int main() {
  auto t = std::make_shared<ThreadPool>(4);
  auto g = std::make_shared<Gomoku>(5, 3, 1);
  auto n = std::make_shared<InstanceNeuralNetwork>();
  g->execute_move(12);
  g->execute_move(13);
  g->display();

  auto m = std::make_shared<MCTS>(t.get(), n.get(), 5, 10000, 0.1,
                                  g->get_action_size());

  std::cout << "RUNNING" << std::endl;

  // auto res = m->get_action_probs(g.get(), 1e-3);
  // std::for_each(res.begin(), res.end(), [](double x) { std::cout << x << ",
  // "; });

  while (g->get_game_status()[0] == 0) {
    auto res = m->get_action_probs(g.get(), 1);
    std::for_each(res.begin(), res.end(),
                  [](const double &x) { std::cout << x << ", "; });
    std::cout << std::endl;

    unsigned int best_move = 0;
    double best_value = -DBL_MAX;

    for (unsigned int i = 0; i < res.size(); i++) {
      if (res[i] > best_value) {
        best_value = res[i];
        best_move = i;
      }
    }
    std::cout << best_move << ", " <<  best_value << std::endl;

    g->execute_move(best_move);
    m->update_with_move(best_move);
    g->display();
  }

  return 0;
}
