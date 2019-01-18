#include <iostream>
#include <thread_pool.h>

int calc(int n) {
  std::this_thread::sleep_for(std::chrono::seconds(n));

  return n;
}

int main() {

  ThreadPool thread_pool(4);

  auto r1 = thread_pool.commit(calc, 1);
  auto r2 = thread_pool.commit(calc, 2);
  auto r3 = thread_pool.commit(calc, 3);
  auto r4 = thread_pool.commit(calc, 4);


  std::cout << r1.get() << std::endl;
  std::cout << r2.get() << std::endl;
  std::cout << r3.get() << std::endl;
  std::cout << r4.get() << std::endl;

  return 0;
}
