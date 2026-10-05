#include "cuda_work1/vector.cuh"

#include "cuda_work1/cuda_check.cuh"

#include <cuda_runtime.h>

#include <stdexcept>

namespace cuda_work1 {

template <typename AtomT>
__global__ void kernel_vecadd(VectorView<AtomT> lhs, VectorView<AtomT> rhs,
                              VectorView<AtomT> result) {
  const std::size_t index = static_cast<std::size_t>(blockIdx.x) *
                                blockDim.x + threadIdx.x;
  if (index < result.size()) {
    result[index] = lhs[index] + rhs[index];
  }
}

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs) {
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument("cuda_work1: vector sizes must match");
  }

  Vector<AtomT> result(lhs.size());
  if (lhs.size() == 0) {
    return result;
  }

  constexpr unsigned int threads_per_block = 256;
  const auto blocks = static_cast<unsigned int>(
      (lhs.size() + threads_per_block - 1) / threads_per_block);

  kernel_vecadd<<<blocks, threads_per_block>>>(lhs.view(), rhs.view(),
                                                result.view());
  CUDA_WORK1_CHECK(cudaGetLastError());
  return result;
}

template Vector<float> operator+(const Vector<float>&, const Vector<float>&);

}  // namespace cuda_work1
