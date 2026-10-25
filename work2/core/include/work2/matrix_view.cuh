#pragma once

#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
  #define WORK2_HOST_DEVICE __host__ __device__
#else
  #define WORK2_HOST_DEVICE
#endif

namespace work2 {

// Невладеющее представление матрицы: указатель + размеры.
// Элементы хранятся построчно (row-major): элемент (i, j) лежит по индексу
// i * ncols + j. Класс тривиально копируемый, поэтому его можно передавать
// в кёрнел по значению.
template <typename AtomT>
class MatrixView {
 public:
  WORK2_HOST_DEVICE MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
      : data_(data)
      , nrows_(nrows)
      , ncols_(ncols) {}

  WORK2_HOST_DEVICE std::size_t size() const {
    return nrows_ * ncols_;
  }

  WORK2_HOST_DEVICE std::size_t nrows() const {
    return nrows_;
  }

  WORK2_HOST_DEVICE std::size_t ncols() const {
    return ncols_;
  }

  WORK2_HOST_DEVICE AtomT& operator[](std::size_t n) {
    return data_[n];
  }

  WORK2_HOST_DEVICE const AtomT& operator[](std::size_t n) const {
    return data_[n];
  }

  WORK2_HOST_DEVICE AtomT& operator()(std::size_t i, std::size_t j) {
    return data_[i * ncols_ + j];
  }

  WORK2_HOST_DEVICE const AtomT& operator()(std::size_t i, std::size_t j) const {
    return data_[i * ncols_ + j];
  }

 private:
  AtomT* data_;
  std::size_t nrows_;
  std::size_t ncols_;
};

static_assert(std::is_trivially_copyable_v<MatrixView<float>>);
static_assert(std::is_trivially_copyable_v<MatrixView<double>>);

}  // namespace work2

#undef WORK2_HOST_DEVICE
