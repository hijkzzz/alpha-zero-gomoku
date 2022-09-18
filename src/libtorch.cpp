#include <libtorch.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <ATen/cuda/CUDAGuard.h>

#include <iostream>

using namespace std::chrono_literals;

NeuralNetwork::NeuralNetwork(std::string model_path, bool use_gpu,
                             unsigned int batch_size)
    : module(std::make_shared<torch::jit::script::Module>(torch::jit::load(model_path.c_str()))),
      use_gpu(use_gpu),
      batch_size(batch_size),
      running(true),
      loop(nullptr) {
  if (this->use_gpu) {
    // move to CUDA
    this->module->to(at::kCUDA);
  }

  // run infer thread
  this->loop = std::make_unique<std::thread>([this] {
    while (this->running) {
      this->infer();
    }
  });
}

NeuralNetwork::~NeuralNetwork() {
  this->running = false;
  this->loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::commit(Gomoku* gomoku) {
  int n = gomoku->get_n();

  // convert data format
  auto board = gomoku->get_board();
  std::vector<int> board0;
  for (unsigned int i = 0; i < board.size(); i++) {
    board0.insert(board0.end(), board[i].begin(), board[i].end());
  }

  torch::Tensor temp =
      torch::from_blob(&board0[0], {1, 1, n, n}, torch::dtype(torch::kInt32));

  torch::Tensor state0 = temp.gt(0).toType(torch::kFloat32);
  torch::Tensor state1 = temp.lt(0).toType(torch::kFloat32);

  int last_move = gomoku->get_last_move();
  int cur_player = gomoku->get_current_color();

  if (cur_player == -1) {
    std::swap(state0, state1);
  }

  torch::Tensor state2 =
      torch::zeros({1, 1, n, n}, torch::dtype(torch::kFloat32));

  if (last_move != -1) {
    state2[0][0][last_move / n][last_move % n] = 1;
  }

  // torch::Tensor states = torch::cat({state0, state1}, 1);
  torch::Tensor states = torch::cat({state0, state1, state2}, 1);

  // emplace task
  std::promise<return_type> promise;
  auto ret = promise.get_future();

  {
    std::lock_guard<std::mutex> lock(this->lock);
    tasks.emplace(std::make_pair(states, std::move(promise)));
  }

  this->cv.notify_all();

  return ret;
}

// TODO: use lock-free queue
// https://github.com/cameron314/concurrentqueue
void NeuralNetwork::infer() {
  // get inputs
  std::vector<torch::Tensor> states;
  std::vector<std::promise<return_type>> promises;

  bool timeout = false;
  while (states.size() < this->batch_size && !timeout) {
    // pop task
    {
      std::unique_lock<std::mutex> lock(this->lock);
      if (this->cv.wait_for(lock, 1ms,
                            [this] { return this->tasks.size() > 0; })) {
        while (states.size() < this->batch_size && this->tasks.size() > 0){
          auto task = std::move(this->tasks.front());
          states.emplace_back(std::move(task.first));
          promises.emplace_back(std::move(task.second));

          this->tasks.pop();
        }

      } else {
        // timeout
        // std::cout << "timeout" << std::endl;
        timeout = true;
      }
    }
  }

  // inputs empty
  if (states.size() == 0) {
    return;
  }

  // infer
  std::vector<torch::jit::IValue> inputs{
      this->use_gpu ? torch::cat(states, 0).to(at::kCUDA)
                    : torch::cat(states, 0)};
  auto result = this->module->forward(inputs).toTuple();

  torch::Tensor p_batch = result->elements()[0]
                              .toTensor()
                              .exp()
                              .toType(torch::kFloat32)
                              .to(at::kCPU);
  torch::Tensor v_batch =
      result->elements()[1].toTensor().toType(torch::kFloat32).to(at::kCPU);

  // set promise value
  for (unsigned int i = 0; i < promises.size(); i++) {
    torch::Tensor p = p_batch[i];
    torch::Tensor v = v_batch[i];

    std::vector<double> prob(static_cast<float*>(p.data_ptr()),
                             static_cast<float*>(p.data_ptr()) + p.size(0));
    std::vector<double> value{v.item<float>()};
    return_type temp{std::move(prob), std::move(value)};

    promises[i].set_value(std::move(temp));
  }
}
