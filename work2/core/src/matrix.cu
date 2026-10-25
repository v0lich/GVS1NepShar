#include "work2/cuda_check.cuh"
#include "work2/matrix.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace work2 {

namespace {

constexpr unsigned int block_side = 16;  // блок 16x16 = 256 нитей

unsigned int ceil_div(std::size_t value, unsigned int divisor) {
  return static_cast<unsigned int>((value + divisor - 1) / divisor);
}

}  // namespace

// Одна нить считает один элемент результата C(row, col).
// x нити -> столбец, y нити -> строка: соседние нити варпа читают соседние
// элементы строки B и пишут соседние элементы строки C (объединённый доступ).
template <typename AtomT>
__global__ void kernel_matmul_naive(
    MatrixView<AtomT> lhs, MatrixView<AtomT> rhs, MatrixView<AtomT> result) {
  const std::size_t row = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t col = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (row >= result.nrows() || col >= result.ncols()) {
    return;
  }

  AtomT sum{};
  for (std::size_t p = 0; p < lhs.ncols(); ++p) {
    sum += lhs(row, p) * rhs(p, col);
  }
  result(row, col) = sum;
}

template <typename AtomT>
void matmul_into(
    const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs, Matrix<AtomT>& result) {
  if (lhs.ncols() != rhs.nrows()) {
    throw std::invalid_argument("work2: inner matrix dimensions must match");
  }
  if (result.nrows() != lhs.nrows() || result.ncols() != rhs.ncols()) {
    throw std::invalid_argument("work2: result matrix has wrong shape");
  }

  if (result.size() == 0) {
    return;
  }

  const dim3 block(block_side, block_side);
  const dim3 grid(ceil_div(result.ncols(), block_side), ceil_div(result.nrows(), block_side));

  kernel_matmul_naive<<<grid, block>>>(lhs.view(), rhs.view(), result.view());
  WORK2_CUDA_CHECK(cudaGetLastError());
}

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs) {
  if (lhs.ncols() != rhs.nrows()) {
    throw std::invalid_argument("work2: inner matrix dimensions must match");
  }

  Matrix<AtomT> result(lhs.nrows(), rhs.ncols());
  matmul_into(lhs, rhs, result);
  return result;
}

template void matmul_into<float>(const Matrix<float>&, const Matrix<float>&, Matrix<float>&);
template void matmul_into<double>(
    const Matrix<double>&, const Matrix<double>&, Matrix<double>&);

template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);

}  // namespace work2
