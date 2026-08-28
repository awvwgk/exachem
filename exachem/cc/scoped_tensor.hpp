/*
 * ExaChem: Open Source Exascale Computational Chemistry Software.
 *
 * Copyright 2023-2024 Pacific Northwest National Laboratory, Battelle Memorial Institute.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include "tamm/tamm.hpp"

#include <utility>
#include <vector>

namespace exachem::cc {

/**
 * @brief RAII scope guard for one or more TAMM tensors.
 *
 * Allocates the referenced tensors on construction (via the variadic
 * @c Tensor::allocate(ExecutionContext*, ...) API) and deallocates them on
 * destruction. This removes the manual, error-prone pattern of pairing an
 * @c allocate() call with a matching @c deallocate() many lines later, which
 * leaks Global Arrays / device memory whenever an early @c return or an
 * exception is hit in between.
 *
 * The guard stores pointers to caller-owned tensors; those tensors must
 * outlive the guard. The guard is move-only and never copies.
 *
 * Usage:
 * @code
 *   tamm::Tensor<T> a{...}, b{...};
 *   exachem::cc::ScopedTensors guard{ec, a, b}; // a, b allocated here
 *   // ... use a, b, may return/throw freely ...
 * @endcode                                       // a, b deallocated here
 *
 * If a tensor must survive the scope, do not place it under the guard, or call
 * @c release() to relinquish deallocation of all held tensors.
 */
template<typename T>
class ScopedTensors {
public:
  template<typename... Tensors>
  explicit ScopedTensors(tamm::ExecutionContext& ec, tamm::Tensor<T>& first, Tensors&... rest) {
    tensors_.reserve(1 + sizeof...(rest));
    tensors_.push_back(&first);
    (tensors_.push_back(&rest), ...);
    tamm::Tensor<T>::allocate(&ec, first, rest...);
  }

  ScopedTensors(const ScopedTensors&)            = delete;
  ScopedTensors& operator=(const ScopedTensors&) = delete;

  ScopedTensors(ScopedTensors&& other) noexcept: tensors_{std::move(other.tensors_)} {
    other.tensors_.clear();
  }

  ScopedTensors& operator=(ScopedTensors&& other) noexcept {
    if(this != &other) {
      deallocate_all();
      tensors_ = std::move(other.tensors_);
      other.tensors_.clear();
    }
    return *this;
  }

  ~ScopedTensors() { deallocate_all(); }

  /// Relinquish deallocation responsibility; the tensors will NOT be freed by
  /// this guard. Use when a tensor must outlive the scope.
  void release() noexcept { tensors_.clear(); }

private:
  void deallocate_all() noexcept {
    for(auto* t: tensors_) t->deallocate();
    tensors_.clear();
  }

  std::vector<tamm::Tensor<T>*> tensors_;
};

// Class template argument deduction: deduce T from the first tensor argument.
template<typename T, typename... Tensors>
ScopedTensors(tamm::ExecutionContext&, tamm::Tensor<T>&, Tensors&...) -> ScopedTensors<T>;

/**
 * @brief Deduction-friendly factory. Allocates @p first and @p rest and returns
 * a guard that deallocates them when it goes out of scope.
 */
template<typename T, typename... Tensors>
[[nodiscard]] ScopedTensors<T> make_scoped_tensors(tamm::ExecutionContext& ec,
                                                   tamm::Tensor<T>& first, Tensors&... rest) {
  return ScopedTensors<T>{ec, first, rest...};
}

} // namespace exachem::cc
