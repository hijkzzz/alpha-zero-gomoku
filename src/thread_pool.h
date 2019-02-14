#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
  using task_type = std::function<void()>;

  inline ThreadPool(unsigned short thread_num = 4) {
    this->run.store(true);
    this->idl_thread_num = thread_num;

    for (unsigned int i = 0; i < thread_num; ++i) {
      // thread type implicit conversion
      pool.emplace_back([this] {
        while (this->run) {
          std::function<void()> task;

          // get a task
          {
            std::unique_lock<std::mutex> lock(this->lock);
            this->cv.wait(lock, [this] {
              return this->tasks.size() > 0 || !this->run.load();
            });

            // exit
            if (!this->run.load() && this->tasks.empty())
              return;

            // pop task
            task = std::move(this->tasks.front());
            this->tasks.pop();
          }

          // run a task
          this->idl_thread_num--;
          task();
          this->idl_thread_num++;
        }
      });
    }
  }

  inline ~ThreadPool() {
    // clean thread pool
    this->run.store(false);
    this->cv.notify_all(); // wake all thread

    for (std::thread &thread : pool) {
      thread.join();
    }
  }

  template <class F, class... Args>
  auto commit(F &&f, Args &&... args) -> std::future<decltype(f(args...))> {
    // commit a task, return std::future
    // example: .commit(std::bind(&Dog::sayHello, &dog));

    if (!this->run.load())
      throw std::runtime_error("commit on ThreadPool is stopped.");

    // declare return type
    using return_type = decltype(f(args...));

    // make a shared ptr for packaged_task
    // packaged_task package the bind function and future
    auto task_ptr = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    {
      std::lock_guard<std::mutex> lock(this->lock);
      tasks.emplace([task_ptr]() { (*task_ptr)(); });
    }

    // wake a thread
    this->cv.notify_one();

    return task_ptr->get_future();
  }

  inline int get_idl_num() { return this->idl_thread_num; }

private:
  std::vector<std::thread> pool; // thead pool
  std::queue<task_type> tasks;   // tasks queue
  std::mutex lock;               // lock for tasks queue
  std::condition_variable cv;    // condition variable for tasks queue

  std::atomic<bool> run;                    // is running
  std::atomic<unsigned int> idl_thread_num; // idle thread number
};
