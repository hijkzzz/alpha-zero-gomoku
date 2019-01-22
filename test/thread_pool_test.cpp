#include <iostream>
#include <thread_pool.h>

unsigned long long calc(unsigned long long n) {

  return n == 0 ? 1 : n * calc(n - 1);
}

int main() {

  ThreadPool thread_pool(4);

  auto r1 = thread_pool.commit(calc, 10);
  auto r2 = thread_pool.commit(calc, 50);
  auto r3 = thread_pool.commit(calc, 100);
  auto r4 = thread_pool.commit(calc, 200);

  std::cout << r1.get() << std::endl;
  std::cout << r2.get() << std::endl;
  std::cout << r3.get() << std::endl;
  std::cout << r4.get() << std::endl;

  std::cout << "END" << std::endl;

  return 0;
}
