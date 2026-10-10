cmake_minimum_required(VERSION 3.16)
project(forge_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FORGE_KERNELS_BUILD_TESTS "Build kernel unit tests" ON)
option(FORGE_KERNELS_ENABLE_AVX2 "Enable AVX2/FMA kernels when supported" ON)
option(FORGE_BUILD_CUDA "Build optional CUDA backend (requires toolkit + nvcc)" OFF)

add_library(forge_kernels STATIC
  src/matmul.cpp
  src/matvec.cpp
  src/linear.cpp
  src/linear_quant.cpp
  src/rmsnorm.cpp
  src/softmax.cpp
  src/rope.cpp
  src/activation.cpp
)

target_include_directories(forge_kernels
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(forge_kernels PUBLIC FORGE_KERNELS_STATIC)

if(MSVC)
  target_compile_options(forge_kernels PRIVATE /W4 /permissive-)
  if(FORGE_KERNELS_ENABLE_AVX2)
    target_compile_options(forge_kernels PRIVATE /arch:AVX2)
  endif()
else()
  target_compile_options(forge_kernels PRIVATE -Wall -Wextra -Wpedantic)
  if(FORGE_KERNELS_ENABLE_AVX2)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx2" FORGE_HAS_MAVX2)
    check_cxx_compiler_flag("-mfma" FORGE_HAS_MFMA)
    if(FORGE_HAS_MAVX2 AND FORGE_HAS_MFMA)
      target_compile_options(forge_kernels PRIVATE -mavx2 -mfma)
    endif()
  endif()
endif()

if(FORGE_KERNELS_BUILD_TESTS)
  enable_testing()
  add_executable(forge_kernels_tests
    tests/test_matmul.cpp
    tests/test_matvec.cpp
    tests/test_ops.cpp
    tests/test_main.cpp
  )
  target_link_libraries(forge_kernels_tests PRIVATE forge_kernels)
  target_include_directories(forge_kernels_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/tests
  )
  add_test(NAME forge_kernels_tests COMMAND forge_kernels_tests)
endif()

add_executable(forge_matmul_bench EXCLUDE_FROM_ALL ../benchmarks/bench_matmul.cpp)
target_link_libraries(forge_matmul_bench PRIVATE forge_kernels)

if(FORGE_BUILD_CUDA)
  enable_language(CUDA)
  add_subdirectory(cuda)
endif()
