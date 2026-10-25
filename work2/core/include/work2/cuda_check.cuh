#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace work2 {

inline void check_cuda(
    cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file + ":"
                             + std::to_string(line) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace work2

#define WORK2_CUDA_CHECK(expr) ::work2::check_cuda((expr), #expr, __FILE__, __LINE__)
