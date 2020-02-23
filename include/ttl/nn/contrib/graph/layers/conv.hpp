#include <tuple>

#include <ttl/nn/ops>

#include <ttl/nn/bits/graph/builder.hpp>

namespace ttl::nn::graph::layers
{
template <typename R, typename builder>
auto cnn(builder &b, const ttl::nn::graph::internal::var_node<R, 4> *x,
         const ttl::shape<2> &filter_shape, int n_filters)
{
    const auto [r, s] = filter_shape.dims();
    const auto [n, h, w, c] = x->shape().dims();
    auto kernel =
        b.template covar<R>("kernel", ttl::make_shape(r, s, c, n_filters),
                            // ttl::nn::ops::truncated_normal(.1)
                            nn::ops::constant<R>(0.1));
    auto bias = b.template covar<R>("bias", ttl::make_shape(n_filters),
                                    ttl::nn::ops::constant<R>(0.1));

    auto y = b.template invoke<R>("conv", ttl::nn::ops::conv<>(), x, kernel);
    auto z = b.template invoke<R>(
        "conv_bias", ttl::nn::ops::add_bias<ttl::nn::ops::nhwc>(), y, bias);

    return std::make_tuple(z, kernel, bias);
}
}  // namespace ttl::nn::graph::layers
