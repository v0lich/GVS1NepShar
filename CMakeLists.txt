cmake_minimum_required(VERSION 3.24)

project(cuda_work1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(FetchContent)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
FetchContent_Declare(
  googlebenchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(eigen googletest googlebenchmark)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_work1_core
  src/vector.cu
)

target_include_directories(cuda_work1_core
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cuda_work1_core
  PUBLIC CUDA::cudart
)

target_compile_features(cuda_work1_core PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(cuda_work1_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
  target_compile_options(cuda_work1_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()

enable_testing()

add_executable(cuda_work1_tests
  tests/vector_add_test.cu
)
target_link_libraries(cuda_work1_tests
  PRIVATE cuda_work1_core GTest::gtest_main Eigen3::Eigen
)
include(GoogleTest)
gtest_discover_tests(cuda_work1_tests)

add_executable(cuda_work1_benchmarks
  benchmarks/vector_add_benchmark.cu
)
target_link_libraries(cuda_work1_benchmarks
  PRIVATE cuda_work1_core benchmark::benchmark Eigen3::Eigen
)

add_custom_target(plot_benchmarks
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/scripts
          python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_benchmarks.py
          ${CMAKE_BINARY_DIR}/benchmarks.json
          ${CMAKE_BINARY_DIR}/plots
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DEPENDS cuda_work1_benchmarks
  COMMENT "Build plots from benchmarks.json"
)
