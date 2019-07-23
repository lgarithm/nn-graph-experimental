#include <ttl/cuda_tensor>
#include <ttl/shape>
#include <ttl/tensor>

namespace example_ops
{
using ttl::shape;

namespace internal
{
template <typename R> struct fill {
    void operator()(const ttl::cuda_tensor_ref<R, 1> &x, R value) const;
};

extern template struct fill<float>;
extern template struct fill<int>;

template <typename R> struct add {
    void operator()(const ttl::cuda_tensor_ref<R, 1> &z,
                    const ttl::cuda_tensor_view<R, 1> &x,
                    const ttl::cuda_tensor_view<R, 1> &y);
};

extern template struct add<float>;
extern template struct add<int>;

}  // namespace internal

class ones
{
  public:
    template <typename R, ttl::rank_t r>
    void operator()(const ttl::cuda_tensor_ref<R, r> &x) const
    {
        internal::fill<R>()(flatten(x), static_cast<R>(1));
    }
};

class add
{
  public:
    template <ttl::rank_t r>
    shape<r> operator()(const shape<r> &x, const shape<r> &y) const
    {
        // contract_assert_eq(x, y);
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::cuda_tensor_ref<R, r> &z,
                    const ttl::cuda_tensor_view<R, r> &x,
                    const ttl::cuda_tensor_view<R, r> &y) const
    {
        internal::add<R>()(flatten(z), flatten(x), flatten(y));
    }
};

}  // namespace example_ops
