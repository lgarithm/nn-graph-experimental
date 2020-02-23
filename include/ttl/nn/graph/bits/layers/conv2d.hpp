#pragma once
#include <ttl/nn/bits/graph/builder.hpp>
#include <ttl/nn/graph/bits/layers/layer.hpp>
#include <ttl/nn/ops>

namespace ttl::nn::graph::layers
{
template <typename image_order = traits::nhwc>
class conv_layer
{
    using conv_op = ops::conv<image_order>;
    using bias_op = ops::add_bias<image_order>;

    const ttl::shape<2> &ksize;
    const int n_filters;

  public:
    conv_layer(const ttl::shape<2> &ksize, int n_filters)
        : ksize(ksize), n_filters(n_filters)
    {
    }

    template <typename R, typename builder, typename Winit = ops::noop,
              typename Binit = ops::noop>
    auto apply(builder &b, const internal::var_node<R, 4> *x,
               const Winit &w_init = Winit(),
               const Binit &b_init = Binit()) const
    {
        const auto [r, s] = ksize.dims();
        const auto [n, h, w, c] = x->shape().dims();
        auto kernel =
            b.template covar<R>("kernel", b.shape(r, s, c, n_filters), w_init);
        auto bias = b.template covar<R>("bias", b.shape(n_filters), b_init);
        auto y = b.template invoke<R>("conv", conv_op(), x, kernel);
        auto z = b.template invoke<R>("conv_bias", bias_op(), y, bias);
        return simple_layer(z, kernel, bias);
    }
};
}  // namespace ttl::nn::graph::layers
