#include "work2/matrix.cuh"
#include "work2/matrix_view.cuh"
#include "work2_tests/eigen_utils.cuh"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>

namespace {

using work2::Matrix;
using work2::MatrixView;
using work2_tests::random_matrix;
using work2_tests::to_device;
using work2_tests::to_host;

constexpr float precision = 1.0e-5F;
constexpr std::array<std::size_t, 7> sizes{1, 2, 3, 127, 128, 129, 512};

// ---------------------------------------------------------------------------
// Сравнение с Eigen: каждая тройка (m, k, n) — отдельный тест.
// A (m x k) * B (k x n) = C (m x n)
// ---------------------------------------------------------------------------

using Shape = std::tuple<std::size_t, std::size_t, std::size_t>;  // m, k, n

class MatmulMatchesEigen : public ::testing::TestWithParam<Shape> {};

TEST_P(MatmulMatchesEigen, ProductEqualsEigenProduct) {
  const auto [m, k, n] = GetParam();

  std::mt19937 gen(42);
  const Eigen::MatrixXf a_host = random_matrix(m, k, gen);
  const Eigen::MatrixXf b_host = random_matrix(k, n, gen);

  const Matrix<float> a = to_device(a_host);
  const Matrix<float> b = to_device(b_host);
  const Matrix<float> c = a * b;

  ASSERT_EQ(c.nrows(), m);
  ASSERT_EQ(c.ncols(), n);

  const Eigen::MatrixXf expected = a_host * b_host;
  const Eigen::MatrixXf actual = to_host(c);

  EXPECT_TRUE(actual.isApprox(expected, precision));
}

std::string shape_name(const ::testing::TestParamInfo<Shape>& info) {
  const auto [m, k, n] = info.param;
  return "m" + std::to_string(m) + "_k" + std::to_string(k) + "_n" + std::to_string(n);
}

INSTANTIATE_TEST_SUITE_P(Sizes,
    MatmulMatchesEigen,
    ::testing::Combine(::testing::ValuesIn(sizes),
        ::testing::ValuesIn(sizes),
        ::testing::ValuesIn(sizes)),
    shape_name);

// ---------------------------------------------------------------------------
// Умножение на заранее посчитанном вручную примере.
// ---------------------------------------------------------------------------

TEST(MatmulKnownValues, SmallRectangularProduct) {
  // A (2x3) = | 1 2 3 |   B (3x2) = |  7  8 |   C = |  58  64 |
  //           | 4 5 6 |             |  9 10 |       | 139 154 |
  //                                 | 11 12 |
  const std::array<float, 6> a_host{1, 2, 3, 4, 5, 6};
  const std::array<float, 6> b_host{7, 8, 9, 10, 11, 12};

  Matrix<float> a(2, 3);
  Matrix<float> b(3, 2);
  a.data().copy_from_host(a_host.data());
  b.data().copy_from_host(b_host.data());

  const Matrix<float> c = a * b;

  std::array<float, 4> c_host{};
  c.data().copy_to_host(c_host.data());

  EXPECT_FLOAT_EQ(c_host[0], 58.0F);
  EXPECT_FLOAT_EQ(c_host[1], 64.0F);
  EXPECT_FLOAT_EQ(c_host[2], 139.0F);
  EXPECT_FLOAT_EQ(c_host[3], 154.0F);
}

// ---------------------------------------------------------------------------
// Граничные случаи и ошибки.
// ---------------------------------------------------------------------------

TEST(MatmulErrors, ThrowsOnInnerDimensionMismatch) {
  const Matrix<float> a(2, 3);
  const Matrix<float> b(4, 2);
  EXPECT_THROW((void)(a * b), std::invalid_argument);
}

TEST(MatmulErrors, MatmulIntoThrowsOnWrongResultShape) {
  const Matrix<float> a(2, 3);
  const Matrix<float> b(3, 4);
  Matrix<float> result(2, 5);
  EXPECT_THROW(work2::matmul_into(a, b, result), std::invalid_argument);
}

TEST(MatmulEdgeCases, EmptyResultIsAllowed) {
  const Matrix<float> a(0, 3);
  const Matrix<float> b(3, 4);

  const Matrix<float> c = a * b;

  EXPECT_EQ(c.nrows(), 0U);
  EXPECT_EQ(c.ncols(), 4U);
  EXPECT_EQ(c.size(), 0U);
}

// ---------------------------------------------------------------------------
// Классы Matrix и MatrixView.
// ---------------------------------------------------------------------------

TEST(MatrixClass, ReportsShapeAndSize) {
  const Matrix<float> m(3, 5);
  EXPECT_EQ(m.nrows(), 3U);
  EXPECT_EQ(m.ncols(), 5U);
  EXPECT_EQ(m.size(), 15U);
  EXPECT_EQ(m.data().size(), 15U);
}

TEST(MatrixClass, CopySharesDeviceData) {
  const Matrix<float> original(2, 2);
  const Matrix<float> copy = original;  // NOLINT(performance-unnecessary-copy-initialization)

  EXPECT_EQ(&copy.data(), &original.data());
  EXPECT_EQ(copy.data().data(), original.data().data());
}

TEST(MatrixView, UsesRowMajorIndexing) {
  // | 0 1 2 |
  // | 3 4 5 |
  std::array<float, 6> storage{0, 1, 2, 3, 4, 5};
  const MatrixView<float> view(storage.data(), 2, 3);

  EXPECT_EQ(view(0, 2), 2.0F);
  EXPECT_EQ(view(1, 0), 3.0F);
  EXPECT_EQ(view(1, 1), view[4]);
}

TEST(MatrixView, IsTriviallyCopyable) {
  EXPECT_TRUE(std::is_trivially_copyable_v<MatrixView<float>>);
}

}  // namespace
