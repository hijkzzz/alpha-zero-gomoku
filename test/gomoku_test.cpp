#include <iostream>
#include <vector>
#include <gomoku.h>

int main() {
  Gomoku gomoku(10, 5);

  // test execute_move
  // gomoku.execute_move(1, std::make_tuple(0, 0));
  // gomoku.execute_move(1, std::make_tuple(1, 1));
  // gomoku.execute_move(1, std::make_tuple(2, 2));
  // gomoku.execute_move(1, std::make_tuple(3, 3));
  // gomoku.execute_move(1, std::make_tuple(4, 4));

  // gomoku.execute_move(1, std::make_tuple(8, 0));
  // gomoku.execute_move(1, std::make_tuple(8, 1));
  // gomoku.execute_move(1, std::make_tuple(8, 2));
  // gomoku.execute_move(1, std::make_tuple(8, 3));
  // gomoku.execute_move(1, std::make_tuple(8, 4));

  // gomoku.execute_move(1, std::make_tuple(0, 9));
  // gomoku.execute_move(1, std::make_tuple(1, 9));
  // gomoku.execute_move(1, std::make_tuple(2, 9));
  // gomoku.execute_move(1, std::make_tuple(3, 9));
  // gomoku.execute_move(1, std::make_tuple(4, 9));

  gomoku.execute_move(-1, std::make_tuple(0, 4));
  gomoku.execute_move(-1, std::make_tuple(1, 3));
  gomoku.execute_move(-1, std::make_tuple(2, 2));
  gomoku.execute_move(-1, std::make_tuple(3, 1));
  // gomoku.execute_move(-1, std::make_tuple(4, 0));

  // test display
  gomoku.display();

  // test get_xxx
  std::cout << gomoku.get_action_size() << std::endl;
  std::cout << gomoku.get_current_color() << std::endl;

  auto last_move = gomoku.get_last_move();
  std::cout << std::get<0>(last_move) << ", " << std::get<1>(last_move)
            << std::endl;

  // test has_legal_moves
  std::cout << gomoku.has_legal_moves() << std::endl;

  // test get_legal_moves
  auto legal_moves = gomoku.get_legal_moves();
  for (unsigned int i = 0; i < legal_moves.size(); i++) {
    std::cout << std::get<0>(legal_moves[i]) << ", " << std::get<1>(legal_moves[i])
              << std::endl;
  }
  std::cout << legal_moves.size() << std::endl;

  // test get_game_status
  auto game_status = gomoku.get_game_status();
  std::cout << std::get<0>(game_status) << ", " << std::get<1>(game_status)
            << std::endl;
}
