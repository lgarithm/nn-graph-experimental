#pragma once
#include <ttl/cuda_tensor>
#include <ttl/tensor>

#include <nn/bits/tuple.hpp>

namespace ttl
{
namespace internal
{
template <typename S> struct super_shape;

template <typename D, rank_t r> struct super_shape<basic_shape<r, D>> {
    using type = basic_shape<r + 1, D>;
};

template <template <typename, rank_t, typename> class T,  //
          typename R, rank_t r, typename S>
struct chunker {
    using dim_t = typename T<R, r, S>::shape_type::dimension_type;
    using S1 = typename super_shape<S>::type;

    T<R, r + 1, S1> operator()(const T<R, r, S> &t, const dim_t &k) const
    {
        static_assert(r > 0, "rank > 0 is requied");
        const auto l = std::get<0>(t.shape().dims());
        const auto n = l / k;
        if (const auto dropped = (l - n * k); dropped > 0) {
            // drop last l - n * k elements
            std::cerr << "dropped last " << dropped << " sub tensors"
                      << std::endl;
        }
        const std::array<dim_t, r - 1> sub_dims = t.shape().subshape().dims();
        const auto dim_tup = std::tuple_cat(std::make_tuple(n, k), sub_dims);
        return T<R, r + 1, S1>(t.data(), tup2arr<dim_t>(dim_tup));
    }
};

template <typename R, rank_t r, typename S>
basic_host_tensor_ref<R, r + 1, typename super_shape<S>::type>
chunk(const basic_host_tensor<R, r, S> &t, int k)
{
    return internal::chunker<basic_host_tensor_ref, R, r, S>()(t, k);
}

template <typename R, rank_t r, typename S>
basic_host_tensor_ref<R, r + 1, typename super_shape<S>::type>
chunk(const basic_host_tensor_ref<R, r, S> &t, int k)
{
    return internal::chunker<basic_host_tensor_ref, R, r, S>()(t, k);
}

template <typename R, rank_t r, typename S>
basic_host_tensor_view<R, r + 1, typename super_shape<S>::type>
chunk(const basic_host_tensor_view<R, r, S> &t, int k)
{
    return internal::chunker<basic_host_tensor_view, R, r, S>()(t, k);
}

template <typename R, rank_t r, typename S>
basic_cuda_tensor_ref<R, r + 1, typename super_shape<S>::type>
chunk(const basic_cuda_tensor<R, r, S> &t, int k)
{
    return internal::chunker<basic_cuda_tensor_ref, R, r, S>()(t, k);
}

template <typename R, rank_t r, typename S>
basic_cuda_tensor_ref<R, r + 1, typename super_shape<S>::type>
chunk(const basic_cuda_tensor_ref<R, r, S> &t, int k)
{
    return internal::chunker<basic_cuda_tensor_ref, R, r, S>()(t, k);
}

template <typename R, rank_t r, typename S>
basic_cuda_tensor_view<R, r + 1, typename super_shape<S>::type>
chunk(const basic_cuda_tensor_view<R, r, S> &t, int k)
{
    return internal::chunker<basic_cuda_tensor_view, R, r, S>()(t, k);
}

}  // namespace internal

using internal::chunk;

}  // namespace ttl
