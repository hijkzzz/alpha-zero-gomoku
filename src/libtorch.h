#pragma once

#include <torch/script.h> // One-stop header.

#include <string>
#include <memory>
#include <vector>

#include <gomoku.h>


class NeuralNetwork {
public:
  NeuralNetwork(std::string model_path, bool use_gpu);
  std::vector<std::vector<double>> infer(Gomoku* gomoku);

private:
  std::shared_ptr<torch::jit::script::Module> module;
  bool use_gpu;
};
