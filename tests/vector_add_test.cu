#include "cuda_work1/vector.cuh"

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace {

using cuda_work1::Vector;

TEST(VectorAdd, MatchesEigenForRequiredSizes) {
  constexpr std::array<std::size_t, 9> sizes{1, 2, 3, 127, 128,
                                              129, 512, 1024, 1029};

  for (const std::size_t n : sizes) {
    SCOPED_TRACE(n);

    std::vector<float> lhs_host(n);
    std::vector<float> rhs_host(n);
    for (std::size_t i = 0; i < n; ++i) {
      lhs_host[i] = 0.25F * static_cast<float>(i) - 3.0F;
      rhs_host[i] = 0.5F * static_cast<float>(i) + 1.0F;
    }

    Vector<float> lhs(n);
    Vector<float> rhs(n);
    lhs.data().copy_from_host(lhs_host.data());
    rhs.data().copy_from_host(rhs_host.data());

    const Vector<float> actual_vector = lhs + rhs;
    std::vector<float> actual_host(n);
    actual_vector.data().copy_to_host(actual_host.data());

    Eigen::VectorXf lhs_eigen = Eigen::Map<const Eigen::VectorXf>(
        lhs_host.data(), static_cast<Eigen::Index>(n));
    Eigen::VectorXf rhs_eigen = Eigen::Map<const Eigen::VectorXf>(
        rhs_host.data(), static_cast<Eigen::Index>(n));
    const Eigen::VectorXf expected = lhs_eigen + rhs_eigen;
    const Eigen::Map<const Eigen::VectorXf> actual(
        actual_host.data(), static_cast<Eigen::Index>(n));

    EXPECT_TRUE(actual.isApprox(expected, 1.0e-6F));

        const float max_absolute_error =
        (actual - expected).cwiseAbs().maxCoeff();

    EXPECT_LE(max_absolute_error, 1.0e-6F);
  }
}


TEST(VectorAdd, RejectsDifferentSizes) {
  Vector<float> lhs(3);
  Vector<float> rhs(4);
  EXPECT_THROW((void)(lhs + rhs), std::invalid_argument);
}
TEST(VectorAdd, HandlesEmptyVectors) {
  Vector<float> lhs(0);
  Vector<float> rhs(0);

  const auto result = lhs + rhs;

  EXPECT_EQ(result.size(), 0u);
  EXPECT_EQ(result.data().data(), nullptr);
}
}  // namespace
