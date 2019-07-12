#pragma once
#include <nn/common.hpp>

namespace nn::ops
{

template <ttl::rank_t r> class reshape_copy
{
    const shape<r> shape_;

  public:
    reshape_copy(const shape<r> &shape) : shape_(shape) {}

    template <ttl::rank_t r1> shape<r> operator()(const shape<r1> &shape) const
    {
        contract_assert_eq(shape_.size(), shape.size());
        return shape_;
    }

    template <typename R, ttl::rank_t r1>
    void operator()(const ttl::tensor_ref<R, r> &y,
                    const ttl::tensor_view<R, r1> &x) const
    {
        std::copy(x.data(), x.data_end(), y.data());
    }
};
}  // namespace nn::ops
