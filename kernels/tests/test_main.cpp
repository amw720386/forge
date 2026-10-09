#include "test_common.hpp"

void test_matmul();
void test_matvec();
void test_ops();

int main() {
  test_matmul();
  test_matvec();
  test_ops();

  if (forge_test::failures() != 0) {
    std::cerr << forge_test::failures() << " test failure(s)\n";
    return 1;
  }
  std::cout << "All forge_kernels tests passed\n";
  return 0;
}
