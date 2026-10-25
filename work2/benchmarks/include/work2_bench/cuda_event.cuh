#pragma once

#include "work2/cuda_check.cuh"

#include <cuda_runtime.h>

namespace work2_bench {

// RAII-обёртка над cudaEvent_t: событие создаётся в конструкторе
// и гарантированно уничтожается в деструкторе.
class CudaEvent {
 public:
  CudaEvent() {
    WORK2_CUDA_CHECK(cudaEventCreate(&event_));
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  ~CudaEvent() noexcept {
    (void)cudaEventDestroy(event_);
  }

  void record() {
    WORK2_CUDA_CHECK(cudaEventRecord(event_));
  }

  void synchronize() {
    WORK2_CUDA_CHECK(cudaEventSynchronize(event_));
  }

  // Время между событиями start и stop в секундах.
  static double elapsed_seconds(const CudaEvent& start, const CudaEvent& stop) {
    float milliseconds = 0.0F;
    WORK2_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start.event_, stop.event_));
    return static_cast<double>(milliseconds) / 1000.0;
  }

 private:
  cudaEvent_t event_{};
};

}  // namespace work2_bench
