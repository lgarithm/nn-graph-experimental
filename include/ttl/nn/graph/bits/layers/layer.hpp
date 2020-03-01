#pragma once
#include <ttl/nn/bits/graph/node.hpp>

namespace ttl::nn::graph
{
template <typename R, rank_t r>
class simple_layer
{
    using output_t = internal::var_node<R, r>;

    const output_t *output_;
    const std::vector<const internal::base_var_node *> params_;

  public:
    template <typename... Args>
    simple_layer(const output_t *x, const Args &... args)
        : output_(x), params_({args...})
    {
    }

    const output_t *operator*() const { return output_; };

    const std::vector<const internal::base_var_node *> &params() const
    {
        return params_;
    }
};
}  // namespace ttl::nn::graph
