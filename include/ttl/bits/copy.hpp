#pragma once
#include <ttl/cuda_tensor>
#include <ttl/tensor>

namespace ttl
{
namespace internal
{
template <typename R, rank_t r>
void copy(const tensor_ref<R, r> &dst, const cuda_tensor_view<R, r> &src)
{
    using copier = internal::cuda_copier;
    copier::copy<copier::d2h>(dst.data(), src.data(), src.data_size());
}

template <typename R, rank_t r>
void copy(const cuda_tensor_ref<R, r> &dst, const tensor_view<R, r> &src)
{
    using copier = internal::cuda_copier;
    copier::copy<copier::h2d>(dst.data(), src.data(), src.data_size());
}

template <typename R> R get(const tensor_view<R, 0> &t) { return t.data()[0]; }

template <typename R> R get(const cuda_tensor_view<R, 0> &t)
{
    ttl::tensor<R, 0> cpu_scalar;
    copy(ref(cpu_scalar), t);
    return get(view(cpu_scalar));
}
}  // namespace internal

using internal::copy;
using internal::get;
}  // namespace ttl
