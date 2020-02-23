#pragma once
#include <memory>
#include <vector>

#include <ttl/nn/bits/graph/symbol.hpp>

namespace ttl::nn::graph::internal
{
class symbol_manager
{
    std::vector<std::unique_ptr<symbol>> vars_;

  public:
    template <typename R, ttl::rank_t r>
    tensor_symbol<R, r> *create(const ttl::shape<r> &shape)
    {
        auto v = new tensor_symbol<R, r>(shape);
        vars_.push_back(std::unique_ptr<symbol>(v));
        return v;
    }

    template <typename R, typename... D>
    tensor_symbol<R, sizeof...(D)> *create(const D &... dims)
    {
        constexpr auto r = sizeof...(D);
        return create<R, r>(ttl::shape<r>(dims...));
    }
};
}  // namespace ttl::nn::graph::internal
