#include "work2/cuda_check.cuh"
#include "work2/matrix.cuh"
#include "work2_bench/cuda_event.cuh"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <vector>

namespace {

using work2_bench::CudaEvent;

void BM_CudaMatmul(benchmark::State& state) {
  const auto n = static_cast<std::size_t>(state.range(0));

  // Выделение и копирование памяти — до цикла, в замер не попадают.
  work2::Matrix<float> a(n, n);
  work2::Matrix<float> b(n, n);
  work2::Matrix<float> c(n, n);

  const std::vector<float> values(n * n, 1.0F);
  a.data().copy_from_host(values.data());
  b.data().copy_from_host(values.data());

  // Прогревочный запуск: первый вызов кёрнела медленнее из-за инициализации.
  work2::matmul_into(a, b, c);
  WORK2_CUDA_CHECK(cudaDeviceSynchronize());

  CudaEvent start;
  CudaEvent stop;

  for (auto _ : state) {
    start.record();
    work2::matmul_into(a, b, c);
    stop.record();
    stop.synchronize();

    state.SetIterationTime(CudaEvent::elapsed_seconds(start, stop));
  }
}

void BM_EigenMatmul(benchmark::State& state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));

  const Eigen::MatrixXf a = Eigen::MatrixXf::Ones(n, n);
  const Eigen::MatrixXf b = Eigen::MatrixXf::Ones(n, n);
  Eigen::MatrixXf c(n, n);

  for (auto _ : state) {
    const auto start = std::chrono::steady_clock::now();
    c.noalias() = a * b;
    const auto stop = std::chrono::steady_clock::now();

    const std::chrono::duration<double> elapsed = stop - start;
    state.SetIterationTime(elapsed.count());
    benchmark::DoNotOptimize(c.data());
    benchmark::ClobberMemory();
  }
}

}  // namespace

// n = 16, 32, 64, 128, 256, 512, 1024
BENCHMARK(BM_CudaMatmul)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_EigenMatmul)
    ->RangeMultiplier(2)
    ->Range(16, 1024)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
