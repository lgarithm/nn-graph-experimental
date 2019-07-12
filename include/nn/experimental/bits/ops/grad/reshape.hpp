#pragma once
#include <nn/bits/ops/constant.hpp>
#include <nn/common.hpp>

namespace nn::experimental::ops::grad
{
template <ttl::rank_t r> class reshape_copy
{
    using F = nn::ops::reshape_copy<r>;
    const F f_;

  public:
    reshape_copy(const F &f) : f_(f) {}

    template <ttl::rank_t r1>
    shape<r> operator()(const shape<r> &gy, const shape<r> &y,
                        const shape<r1> &x) const
    {
        return nn::ops::gradient_shape<0>(f_, gy, y, x);
    }

    template <typename R, ttl::rank_t r1>
    void operator()(const ttl::tensor_ref<R, r1> &gx,
                    const ttl::tensor_view<R, r> &gy,
                    const ttl::tensor_view<R, r> &y,
                    const ttl::tensor_view<R, r1> &x) const
    {
        std::copy(gy.data(), gy.data_end(), gx.data());
    }
};
}  // namespace nn::experimental::ops::grad
