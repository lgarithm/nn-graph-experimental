#pragma once
#include <ttl/bits/std_cuda_tensor.hpp>
#include <ttl/bits/std_host_tensor.hpp>
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

template <typename R, typename S, typename D, typename A> struct chunker {
    using S1 = typename super_shape<S>::type;
    using A1 = typename basic_tensor_traits<R, A, D>::Access;
    using T1 = basic_tensor<R, S1, D, A1>;
    using T = basic_tensor<R, S, D, A>;
    using dim_t = typename T::shape_type::dimension_type;

    static constexpr rank_t r = S::rank;

    T1 operator()(const T &t, const dim_t &k) const
    {
        static_assert(r > 0, "rank > 0 is requied");
        const dim_t l = std::get<0>(t.shape().dims());
        const dim_t n = l / k;
        if (const dim_t dropped = (l - n * k); dropped > 0) {
            // drop last l - n * k elements
            std::cerr << "dropped last " << dropped << " sub tensors"
                      << std::endl;
        }
        const std::array<dim_t, r - 1> sub_dims = t.shape().subshape().dims();
        const auto dim_tup = std::tuple_cat(std::make_tuple(n, k), sub_dims);
        return T1(t.data(), tup2arr<dim_t>(dim_tup));
    }
};

template <typename R, typename S, typename D, typename A>
typename chunker<R, S, D, A>::T1 chunk(const basic_tensor<R, S, D, A> &t, int k)
{
    return chunker<R, S, D, A>()(t, k);
}
}  // namespace internal

using internal::chunk;
}  // namespace ttl
