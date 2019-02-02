#include <libtorch.h>

#include <iostream>

NeuralNetwork::NeuralNetwork(std::string model_path)
    : module(torch::jit::load(model_path.c_str())) {
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
      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt32))
          .to(at::kCUDA)
          .toType(torch::kFloat32);

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
  auto result = this->module->forward(inputs).toTuple();

  torch::Tensor p = result->elements()[0]
                        .toTensor()
                        .exp()
                        .toType(torch::kFloat32)
                        .to(at::kCPU)[0];
  torch::Tensor v =
      result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU)[0];

  // debug
  std::cout << p << std::endl;
  std::cout << v << std::endl;

  // output
  std::vector<double> prob(static_cast<float*>(p.data_ptr()),
                           static_cast<float*>(p.data_ptr()) + n * n);
  std::vector<double> value{v.item<float>()};

  return {prob, value};
}
