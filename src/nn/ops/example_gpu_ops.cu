#include <ttl/cuda_tensor>
#include <ttl/shape>

#include <thrust/device_vector.h>

#include <nn/bits/ops/example_gpu_ops.hpp>

namespace example_ops
{
namespace
{
template <template <typename, ttl::rank_t, typename> class T, typename R,
          ttl::rank_t r, typename S>
thrust::device_ptr<R> begin(const T<R, r, S> &t)
{
    R *d = const_cast<R *>(t.data());
    return thrust::device_pointer_cast(d);
}

template <template <typename, ttl::rank_t, typename> class T, typename R,
          ttl::rank_t r, typename S>
thrust::device_ptr<R> end(const T<R, r, S> &t)
{
    R *d = const_cast<R *>(t.data_end());
    return thrust::device_pointer_cast(d);
}
}  // namespace

namespace internal
{
template <typename R>
void fill<R>::operator()(const ttl::cuda_tensor_ref<R, 1> &x, R value) const
{
    thrust::fill(begin(x), end(x), value);
}

template struct fill<float>;
template struct fill<int>;

template <typename R>
void add<R>::operator()(const ttl::cuda_tensor_ref<R, 1> &z,
                        const ttl::cuda_tensor_view<R, 1> &x,
                        const ttl::cuda_tensor_view<R, 1> &y)
{
    thrust::transform(begin(x), end(x), begin(y), begin(z), thrust::plus<R>());
}

template struct add<float>;
template struct add<int>;

}  // namespace internal
}  // namespace example_ops
