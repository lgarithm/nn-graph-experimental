#include <ttl/cuda_tensor>
#include <ttl/shape>

namespace example_ops
{

namespace internal
{
template <typename R> struct add {
    void operator()(const ttl::cuda_tensor_ref<R, 1> &z,
                    const ttl::cuda_tensor_view<R, 1> &x,
                    const ttl::cuda_tensor_view<R, 1> &y);
};

extern template struct add<float>;

}  // namespace internal

class add
{
  public:
    template <typename R, ttl::rank_t r>
    void operator()(const ttl::cuda_tensor_ref<R, r> &z,
                    const ttl::cuda_tensor_view<R, r> &x,
                    const ttl::cuda_tensor_view<R, r> &y) const
    {
        internal::add<R>()(flatten(z), flatten(x), flatten(y));
    }
};

}  // namespace example_ops
