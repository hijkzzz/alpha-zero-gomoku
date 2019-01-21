#include <math.h>
#include <gomoku.h>

#include <iostream>

Gomoku::Gomoku(unsigned int n, unsigned int n_in_row, int first_color)
    : n(n), n_in_row(n_in_row), cur_color(first_color), last_move(std::make_tuple(-1, -1)) {
  for (unsigned int i = 0; i < n; i++) {
    this->board.emplace_back(std::vector<int>(n, 0));
  }
};

std::vector<int> Gomoku::get_legal_moves() {
  auto n = this->n;
  std::vector<int> legal_moves(this->get_action_size());

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        legal_moves[i * this->n + j] = 1;
      }
    }
  }

  return legal_moves;
};

bool Gomoku::has_legal_moves() {
  auto n = this->n;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        return true;
      }
    }
  }
  return false;
};

void Gomoku::execute_move(const move_type &move) {
  auto i = move / this->get_n();
  auto j = move % this->get_n();

  if (!this->board[i][j] == 0) {
    throw std::runtime_error("execute_move borad[i][j] != 0.");
  }

  this->board[i][j] = this->cur_color;
  this->last_move = move;
  this->cur_color = -this->cur_color;
};

std::tuple<bool, int> Gomoku::get_game_status() {
  // return (is ended, winner)
  auto n = this->n;
  auto n_in_row = this->n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        continue;
      }

      if (j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i][j + k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row && j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j + k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row && j >= n_in_row - 1) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j - k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }
    }
  }

  if (this->has_legal_moves()) {
    return std::make_tuple(false, 0);
  } else {
    return std::make_tuple(true, 0);
  }
};


void Gomoku::display() {
  auto n = this->board.size();

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      std::cout << this->board[i][j] << ", ";
    }
    std::cout << std::endl;
  }
}
