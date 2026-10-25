#include "cuda_work1/vector.cuh"

#include <benchmark/benchmark.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include "cuda_work1/cuda_check.cuh"

#include <chrono>
#include <cstddef>
#include <vector>

namespace {

void BM_CudaVectorAdd(benchmark::State& state) {
  const auto n =
      static_cast<std::size_t>(state.range(0));

  cuda_work1::Vector<float> lhs(n);
  cuda_work1::Vector<float> rhs(n);
  cuda_work1::Vector<float> result(n);

  std::vector<float> values(n, 1.0F);
  lhs.data().copy_from_host(values.data());
  rhs.data().copy_from_host(values.data());

  // Первый запуск и ожидание — вне измерения.
  cuda_work1::vector_add_into(lhs, rhs, result);
  CUDA_WORK1_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{};
  cudaEvent_t stop{};
  CUDA_WORK1_CHECK(cudaEventCreate(&start));
  CUDA_WORK1_CHECK(cudaEventCreate(&stop));

  for (auto _ : state) {
    CUDA_WORK1_CHECK(cudaEventRecord(start));

    cuda_work1::vector_add_into(lhs, rhs, result);

    CUDA_WORK1_CHECK(cudaEventRecord(stop));
    CUDA_WORK1_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0F;
    CUDA_WORK1_CHECK(
        cudaEventElapsedTime(&milliseconds, start, stop));

    state.SetIterationTime(
        static_cast<double>(milliseconds) / 1000.0);

    benchmark::DoNotOptimize(result.data().data());
  }

  CUDA_WORK1_CHECK(cudaEventDestroy(start));
  CUDA_WORK1_CHECK(cudaEventDestroy(stop));
}

void BM_EigenVectorAdd(benchmark::State& state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  const Eigen::VectorXf lhs = Eigen::VectorXf::Ones(n);
  const Eigen::VectorXf rhs = Eigen::VectorXf::Ones(n);
  Eigen::VectorXf result(n);

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    result.noalias() = lhs + rhs;
    const auto stop = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed = stop - start;
    state.SetIterationTime(elapsed.count());
    benchmark::DoNotOptimize(result.data());
  }
}

}  // namespace

BENCHMARK(BM_CudaVectorAdd)
    ->Args({1 << 3})
    ->Args({1 << 6})
    ->Args({1 << 9})
    ->Args({1 << 12})
    ->Args({1 << 15})
    ->Args({1 << 18})
    ->Args({1 << 21})
    ->Args({1 << 24})
    ->UseManualTime();

BENCHMARK(BM_EigenVectorAdd)
    ->Args({1 << 3})
    ->Args({1 << 6})
    ->Args({1 << 9})
    ->Args({1 << 12})
    ->Args({1 << 15})
    ->Args({1 << 18})
    ->Args({1 << 21})
    ->Args({1 << 24})
    ->UseManualTime();

BENCHMARK_MAIN();
