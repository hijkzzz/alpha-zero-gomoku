#include <iostream>
#include <vector>
#include <gomoku.h>

int main() {
  Gomoku gomoku(10, 5, 1);

  // test execute_move
  gomoku.execute_move(3);
  gomoku.execute_move(4);
  gomoku.execute_move(6);
  gomoku.execute_move(23);
  gomoku.execute_move(8);
  gomoku.execute_move(9);
  gomoku.execute_move(78);
  gomoku.execute_move(0);
  gomoku.execute_move(17);
  gomoku.execute_move(7);
  gomoku.execute_move(19);
  gomoku.execute_move(67);
  gomoku.execute_move(60);
  gomoku.execute_move(14);
  gomoku.execute_move(11);
  gomoku.execute_move(2);
  gomoku.execute_move(99);
  gomoku.execute_move(10);
  gomoku.execute_move(1);
  gomoku.execute_move(5);
  gomoku.execute_move(18);
  gomoku.execute_move(12);
  gomoku.execute_move(15);
  gomoku.execute_move(24);
  gomoku.execute_move(16);

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
  std::cout << std::endl;

  // test get_game_status
  auto game_status = gomoku.get_game_status();
  std::cout << game_status[0] << ", " << game_status[1] << std::endl;
}
