cmake_minimum_required(VERSION 3.18)
project(forge_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to widely available arch; override with -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
endif()

option(FORGE_CUDA_BUILD_TESTS "Build CUDA unit tests (requires GPU)" ON)
option(FORGE_CUDA_BUILD_BENCH "Build CUDA microbenchmarks" ON)

find_package(CUDAToolkit REQUIRED)

add_library(forge_cuda STATIC
  src/device.cu
  src/memory.cu
  src/matmul.cu
  src/matvec.cu
  src/rmsnorm.cu
  src/softmax.cu
  src/activation.cu
  src/linear.cu
  src/rope.cu
  src/embedding.cu
  src/attention.cu
  src/elem.cu
  src/argmax.cu
)

target_include_directories(forge_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(forge_cuda PUBLIC FORGE_CUDA_STATIC)
target_compile_options(forge_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>)
target_link_libraries(forge_cuda PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt)
# OFF: static lib linked from Rust/cargo must not require a separate nvcc device-link step
# (CUDA_SEPARABLE_COMPILATION ON leaves __cudaRegisterLinkedBinary_* unresolved under rust-lld).
set_target_properties(forge_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if(FORGE_CUDA_BUILD_TESTS)
  enable_testing()
  add_executable(forge_cuda_tests
    tests/test_main.cpp
    tests/test_cuda_ops.cpp
  )
  target_link_libraries(forge_cuda_tests PRIVATE forge_cuda)
  target_include_directories(forge_cuda_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests
  )
  # CPU reference kernels for parity (optional link when parent built).
  if(TARGET forge_kernels)
    target_link_libraries(forge_cuda_tests PRIVATE forge_kernels)
    target_compile_definitions(forge_cuda_tests PRIVATE FORGE_CUDA_HAS_CPU_REF=1)
  endif()
  add_test(NAME forge_cuda_tests COMMAND forge_cuda_tests)
endif()

if(FORGE_CUDA_BUILD_BENCH)
  add_executable(forge_cuda_bench bench/bench_cuda.cpp)
  target_link_libraries(forge_cuda_bench PRIVATE forge_cuda)
  target_include_directories(forge_cuda_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
endif()
