
#if !defined(__GOMOKU__)
#define __GOMOKU__

#include <tuple>
#include <vector>

class Gomoku {
public:
  Gomoku(unsigned int n, unsigned int n_in_row);

  std::vector<std::tuple<unsigned int, unsigned int>> get_legal_moves();
  bool has_legal_moves();
  void execute_move(int color,
                    const std::tuple<unsigned int, unsigned int> &move);

  const std::vector<std::vector<int>> &get_board();
  std::tuple<bool, int> get_game_status();

private:
  std::vector<std::vector<int>> board;
  unsigned int n;
  unsigned int n_in_row;
};

#endif // __GOMOKU__
