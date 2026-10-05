#pragma once

#include "cuda_work1/data.cuh"
#include "cuda_work1/vector_view.cuh"

#include <cstddef>
#include <memory>

namespace cuda_work1 {

template <typename AtomT>
class Vector {
 public:
  explicit Vector(std::size_t size)
      : data_(std::make_shared<Data<AtomT>>(size)),
        view_(data_->data(), size) {}

  std::size_t size() const noexcept { return data_->size(); }

  Data<AtomT>& data() noexcept { return *data_; }
  const Data<AtomT>& data() const noexcept { return *data_; }

  VectorView<AtomT>& view() noexcept { return view_; }
  const VectorView<AtomT>& view() const noexcept { return view_; }

 private:
  std::shared_ptr<Data<AtomT>> data_;
  VectorView<AtomT> view_;
};

template <typename AtomT>
void vector_add_into(const Vector<AtomT>& lhs,
                     const Vector<AtomT>& rhs,
                     Vector<AtomT>& result);

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& lhs, const Vector<AtomT>& rhs);

}  // namespace cuda_work1
