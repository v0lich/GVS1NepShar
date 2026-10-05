#pragma once

#include <cstddef>
#include <type_traits>

#if defined(__CUDACC__)
#define CUDA_WORK1_HOST_DEVICE __host__ __device__
#else
#define CUDA_WORK1_HOST_DEVICE
#endif

namespace cuda_work1 {

template <typename AtomT>
class VectorView {
 public:
  CUDA_WORK1_HOST_DEVICE VectorView(AtomT* data, std::size_t size)
      : data_(data), size_(size) {}

  CUDA_WORK1_HOST_DEVICE std::size_t size() const { return size_; }

  CUDA_WORK1_HOST_DEVICE AtomT& operator[](std::size_t index) {
    return data_[index];
  }

  CUDA_WORK1_HOST_DEVICE const AtomT& operator[](std::size_t index) const {
    return data_[index];
  }

  CUDA_WORK1_HOST_DEVICE AtomT& operator()(std::size_t index) {
    return data_[index];
  }

  CUDA_WORK1_HOST_DEVICE const AtomT& operator()(std::size_t index) const {
    return data_[index];
  }

 private:
  AtomT* data_;
  std::size_t size_;
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>);

}  // namespace cuda_work1

#undef CUDA_WORK1_HOST_DEVICE
