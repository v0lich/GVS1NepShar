#pragma once

#include "cuda_work1/cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace cuda_work1 {

template <typename AtomT>
class Data {
  static_assert(std::is_trivially_copyable_v<AtomT>,
                "Data requires a trivially copyable element type");

 public:
  explicit Data(std::size_t size) : size_(size), data_(nullptr) {
    if (size_ != 0) {
      CUDA_WORK1_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  size_ * sizeof(AtomT)));
    }
  }

  Data(const Data& other) : Data(other.size_) {
    if (size_ != 0) {
      CUDA_WORK1_CHECK(cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT),
                                  cudaMemcpyDeviceToDevice));
    }
  }

  Data(Data&& other) noexcept
      : size_(std::exchange(other.size_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

  Data& operator=(const Data& other) {
    if (this != &other) {
      Data copy(other);
      swap(copy);
    }
    return *this;
  }

  Data& operator=(Data&& other) noexcept {
    if (this != &other) {
      release();
      size_ = std::exchange(other.size_, 0);
      data_ = std::exchange(other.data_, nullptr);
    }
    return *this;
  }

  ~Data() noexcept { release(); }

  AtomT* data() noexcept { return data_; }
  const AtomT* data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

  void copy_to_host(AtomT* host_ptr) const {
    if (size_ != 0) {
      CUDA_WORK1_CHECK(cudaMemcpy(host_ptr, data_, size_ * sizeof(AtomT),
                                  cudaMemcpyDeviceToHost));
    }
  }

  void copy_from_host(const AtomT* host_ptr) {
    if (size_ != 0) {
      CUDA_WORK1_CHECK(cudaMemcpy(data_, host_ptr, size_ * sizeof(AtomT),
                                  cudaMemcpyHostToDevice));
    }
  }

  void swap(Data& other) noexcept {
    std::swap(size_, other.size_);
    std::swap(data_, other.data_);
  }

 private:
  void release() noexcept {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
      data_ = nullptr;
    }
    size_ = 0;
  }

  std::size_t size_;
  AtomT* data_;
};

}  // namespace cuda_work1
