
#pragma once
#include <ttl/nn/bits/graph/builder.hpp>
#include <ttl/nn/graph/bits/layers/layer.hpp>
#include <ttl/nn/ops>

namespace ttl::nn::graph::layers
{
class dense_layer
{
    const int logits;

  public:
    dense_layer(const int n) : logits(n) {}

    template <typename R, typename builder, typename Winit = ops::noop,
              typename Binit = ops::noop>
    auto apply(builder &b, const internal::var_node<R, 2> *x,
               const Winit &w_init = Winit(),
               const Binit &b_init = Binit()) const
    {
        auto weight = b.template covar<R>(
            "w", make_shape(std::get<1>(x->shape().dims()), logits), w_init);
        auto bias = b.template covar<R>("b", make_shape(logits), b_init);
        auto y = b.template invoke<R>("prod", ops::matmul(), x, weight);
        auto z = b.template invoke<R>("biased", ops::add_bias<traits::hw>(), y,
                                      bias);
        return simple_layer(z, weight, bias);
    }
};
}  // namespace ttl::nn::graph::layers
