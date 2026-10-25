#pragma once

#include "work2/matrix.cuh"

#include <Eigen/Dense>

#include <cstddef>
#include <random>

namespace work2_tests {

// Eigen по умолчанию хранит матрицы по столбцам (column-major), а наша
// Matrix — по строкам (row-major). Через этот тип делаем перестановку.
using RowMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

inline Eigen::MatrixXf random_matrix(std::size_t nrows, std::size_t ncols, std::mt19937& gen) {
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  Eigen::MatrixXf result(
      static_cast<Eigen::Index>(nrows), static_cast<Eigen::Index>(ncols));
  for (Eigen::Index i = 0; i < result.rows(); ++i) {
    for (Eigen::Index j = 0; j < result.cols(); ++j) {
      result(i, j) = dist(gen);
    }
  }
  return result;
}

inline work2::Matrix<float> to_device(const Eigen::MatrixXf& host) {
  const RowMajorMatrixXf row_major = host;
  work2::Matrix<float> device(
      static_cast<std::size_t>(host.rows()), static_cast<std::size_t>(host.cols()));
  device.data().copy_from_host(row_major.data());
  return device;
}

inline Eigen::MatrixXf to_host(const work2::Matrix<float>& device) {
  RowMajorMatrixXf row_major(static_cast<Eigen::Index>(device.nrows()),
      static_cast<Eigen::Index>(device.ncols()));
  device.data().copy_to_host(row_major.data());
  return row_major;
}

}  // namespace work2_tests
