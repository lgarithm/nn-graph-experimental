#pragma once
#include <ttl/cuda_tensor>
#include <ttl/experimental/copy>
#include <ttl/tensor>

namespace ttl
{
namespace internal
{
template <typename R>
R get(const tensor_view<R, 0> &t)
{
    return t.data()[0];
}

template <typename R>
R get(const cuda_tensor_view<R, 0> &t)
{
    ttl::tensor<R, 0> cpu_scalar;
    copy(ref(cpu_scalar), t);
    return get(view(cpu_scalar));
}
}  // namespace internal

using internal::get;
}  // namespace ttl
