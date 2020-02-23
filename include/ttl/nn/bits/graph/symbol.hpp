#pragma once
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/shape>

namespace ttl::nn::graph::internal
{
template <typename R, ttl::rank_t r>
class tensor_symbol;

class symbol
{
  public:
    virtual ~symbol() {}

    template <typename R, ttl::rank_t r>
    tensor_symbol<R, r> *as() const
    {
        using T = tensor_symbol<R, r>;
        return const_cast<T *>(down_cast<T>(this));
    }
};

template <typename R, ttl::rank_t r>
class tensor_symbol : public symbol
{
    const ttl::shape<r> shape_;

  public:
    tensor_symbol(const tensor_symbol &) = delete;

    tensor_symbol(const ttl::shape<r> &shape) : shape_(shape) {}

    ttl::shape<r> shape() const { return shape_; }
};

}  // namespace ttl::nn::graph::internal
