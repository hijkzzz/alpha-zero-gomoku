#pragma once

#include <tuple>
#include <vector>

class Gomoku {
public:
  using move_type = int;
  using board_type = std::vector<std::vector<int>>;

  Gomoku(unsigned int n, unsigned int n_in_row, int first_color);

  bool has_legal_moves();
  std::vector<int> get_legal_moves();
  void execute_move(move_type move);
  std::vector<int> get_game_status();
  void display() const;

  inline unsigned int get_action_size() const { return this->n * this->n; }
  inline board_type get_board() const { return this->board; }
  inline move_type get_last_move() const { return this->last_move; }
  inline int get_current_color() const { return this->cur_color; }
  inline unsigned int get_n() const { return this->n; }

private:
  board_type board;      // game borad
  unsigned int n;        // board size
  unsigned int n_in_row; // 5 in row or else

  int cur_color;       // current player's color
  move_type last_move; // last move
};
