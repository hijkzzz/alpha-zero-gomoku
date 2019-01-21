#include <iostream>
#include <vector>
#include <gomoku.h>

int main() {
  Gomoku gomoku(10, 5, 1);

  // test execute_move
  gomoku.execute_move(10 * 0 + 4);
  gomoku.execute_move(10 * 6 + 1);

  gomoku.execute_move(10 * 1 + 3);
  gomoku.execute_move(10 * 6 + 2);

  gomoku.execute_move(10 * 2 + 2);
  gomoku.execute_move(10 * 6 + 3);

  gomoku.execute_move(10 * 3 + 1);
  gomoku.execute_move(10 * 6 + 4);

  gomoku.execute_move(10 * 4 + 0);

  // test display
  gomoku.display();

  // test get_xxx
  std::cout << gomoku.get_action_size() << std::endl;
  std::cout << gomoku.get_current_color() << std::endl;

  std::cout << gomoku.get_last_move() << std::endl;

  // test has_legal_moves
  std::cout << gomoku.has_legal_moves() << std::endl;

  // test get_legal_moves
  auto legal_moves = gomoku.get_legal_moves();
  for (unsigned int i = 0; i < legal_moves.size(); i++) {
    std::cout << legal_moves[i] << ", ";
  }
  std::cout << legal_moves.size() << std::endl;

  // test get_game_status
  auto game_status = gomoku.get_game_status();
  std::cout << std::get<0>(game_status) << ", " << std::get<1>(game_status)
            << std::endl;
}
