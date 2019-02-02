#include <iostream>
#include <mcts.h>

int main() {
  // auto t = std::make_shared<ThreadPool>(4);
  // auto g = std::make_shared<Gomoku>(10, 5, 1);
  // auto n = std::make_shared<TestNeuralNetwork>();
  // g->execute_move(12);
  // g->execute_move(13);
  // g->execute_move(14);
  // g->execute_move(15);
  // g->execute_move(16);
  // g->execute_move(17);
  // g->execute_move(18);
  // g->execute_move(19);
  // g->display();

  // auto m = std::make_shared<MCTS>(t.get(), n.get(), 5, 400, 0.5,
  //                                 g->get_action_size());

  // std::cout << "RUNNING" << std::endl;

  // auto res = m->get_action_probs(g.get(), 1);
  // std::for_each(res.begin(), res.end(),
  //               [](double x) { std::cout << x << ", "; });

  // while (g->get_game_status()[0] == 0) {
  //   auto res = m->get_action_probs(g.get(), 1);
  //   std::for_each(res.begin(), res.end(),
  //                 [](const double &x) { std::cout << x << ", "; });
  //   std::cout << std::endl;

  //   unsigned int best_move = 0;
  //   double best_value = -DBL_MAX;

  //   for (unsigned int i = 0; i < res.size(); i++) {
  //     if (res[i] > best_value) {
  //       best_value = res[i];
  //       best_move = i;
  //     }
  //   }
  //   std::cout << best_move << ", " <<  best_value << std::endl;

  //   g->execute_move(best_move);
  //   m->update_with_move(best_move);
  //   g->display();
  // }

  // memory test

  // while (true) {
  //   auto res = m->get_action_probs(g.get(), 1);
  //   std::for_each(res.begin(), res.end(),
  //                 [](double x) { std::cout << x << ","; });
  //   std::cout << std::endl;
  //   m->update_with_move(-1);

  //   break;
  // }

  return 0;
}
