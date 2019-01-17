#include <iostream>
#include <vector>
#include <gomoku.h>

void display(std::vector<std::vector<int>> board) {
  auto n = board.size();

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      std::cout << board[i][j] << " ,";
    }
    std::cout << std::endl;
  }
}

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

  display(gomoku.get_board());

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
