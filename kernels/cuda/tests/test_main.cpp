#include <cstdio>
#include <cstdlib>

#include "forge_cuda.h"

void test_cuda_ops();

int main() {
  if (!forge_cuda_is_available()) {
    std::printf("SKIP: no CUDA device (forge_cuda_is_available=0)\n");
    return 0;
  }
  char name[256] = {};
  forge_cuda_device_name(0, name, sizeof(name));
  std::printf("CUDA device 0: %s\n", name);
  std::printf("forge_cuda: %s\n", forge_cuda_version());

  test_cuda_ops();
  std::printf("All forge_cuda tests passed\n");
  return 0;
}
