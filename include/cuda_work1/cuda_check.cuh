#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace cuda_work1 {

inline void check_cuda(cudaError_t status, const char* expression,
                       const char* file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + " failed at " + file +
                             ":" + std::to_string(line) + ": " +
                             cudaGetErrorString(status));
  }
}

}  // namespace cuda_work1

#define CUDA_WORK1_CHECK(expr) \
  ::cuda_work1::check_cuda((expr), #expr, __FILE__, __LINE__)
