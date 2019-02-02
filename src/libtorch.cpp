#include <libtorch.h>

#include <iostream>

NeuralNetwork::NeuralNetwork(std::string model_path)
    : module(torch::jit::load(model_path)) {
  // move to CUDA
  this->module->to(at::kCUDA);
}

std::vector<std::vector<double>> NeuralNetwork::infer(Gomoku* gomoku) {
  int n = gomoku->get_n();

  // convert data format
  auto board = gomoku->get_board();
  std::vector<int> board0;
  for (unsigned int i = 0; i < board.size(); i++) {
    board0.insert(board0.end(), board[i].begin(), board[i].end());
  }

  torch::Tensor temp =
      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt64))
          .toType(torch::kFloat32)
          .to(at::kCUDA);

  torch::Tensor state0 = temp.gt(0);
  torch::Tensor state1 = temp.lt(0);

  // debug
  std::cout << state0 << std::endl;
  std::cout << state1 << std::endl;

  int last_move = gomoku->get_last_move();
  int cur_player = gomoku->get_current_color();

  torch::Tensor state2 =
      torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
  if (last_move != -1) {
    state2[0][0][last_move / n][last_move % n] = 1;
  }
  torch::Tensor state3 =
      torch::ones({1, 1, n, n}, torch::dtype(torch::kFloat32)).to(at::kCUDA);
  state3 *= cur_player;

  // debug
  std::cout << state2 << std::endl;
  std::cout << state3 << std::endl;

  // infer
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::cat({state0, state1, state2, state3}, 1));
  torch::Tensor result = this->module->forward(inputs).toTensor().to(at::kCPU);

  std::cout << result << std::endl;

  // output
  return {{}};
}
