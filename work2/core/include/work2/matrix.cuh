#pragma once

#include "work2/data.cuh"
#include "work2/matrix_view.cuh"

#include <cstddef>
#include <memory>

namespace work2 {

// Фасад: владеет данными через shared_ptr (агрегация с Data) и хранит
// собственное представление MatrixView (композиция).
template <typename AtomT>
class Matrix {
 public:
  Matrix(std::size_t nrows, std::size_t ncols)
      : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
      , view_(data_->data(), nrows, ncols) {}

  std::size_t size() const noexcept {
    return view_.size();
  }

  std::size_t nrows() const noexcept {
    return view_.nrows();
  }

  std::size_t ncols() const noexcept {
    return view_.ncols();
  }

  Data<AtomT>& data() noexcept {
    return *data_;
  }

  const Data<AtomT>& data() const noexcept {
    return *data_;
  }

  MatrixView<AtomT>& view() noexcept {
    return view_;
  }

  const MatrixView<AtomT>& view() const noexcept {
    return view_;
  }

 private:
  // Порядок полей важен: view_ инициализируется указателем из data_.
  std::shared_ptr<Data<AtomT>> data_;
  MatrixView<AtomT> view_;
};

// result = lhs * rhs без выделения памяти (используется в бенчмарках).
template <typename AtomT>
void matmul_into(
    const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs, Matrix<AtomT>& result);

template <typename AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& lhs, const Matrix<AtomT>& rhs);

}  // namespace work2
