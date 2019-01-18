
#if !defined(__GOMOKU__)
#define __GOMOKU__

#include <tuple>
#include <vector>

class Gomoku {
public:
  using move_type = std::tuple<unsigned int, unsigned int>;
  using board_type = std::vector<std::vector<int>>;

  Gomoku(unsigned int n, unsigned int n_in_row, int first_color);

  bool has_legal_moves();
  std::vector<move_type> get_legal_moves();
  void execute_move(const move_type &move);
  std::tuple<bool, int> get_game_status();
  void display();

  inline unsigned int get_action_size() { return this->n * this->n; }
  inline const board_type &get_board() { return this->board; }
  inline move_type get_last_move() { return this->last_move; }
  inline int get_current_color() { return this->cur_color; }
  inline unsigned int get_n() { return this->n; }

private:
  board_type board;      // game borad
  unsigned int n;        // board size
  unsigned int n_in_row; // 5 in row or else

  int cur_color;       // current player's color
  move_type last_move; // last move
};

#endif // __GOMOKU__
