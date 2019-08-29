#include <tuple>

#include <nn/ops>

#include <nn/bits/graph/builder.hpp>

namespace nn::graph::layers
{
template <typename R, typename builder>
auto cnn(builder &b, const nn::graph::internal::var_node<R, 4> *x,
         const ttl::shape<2> &filter_shape, int n_filters)
{
    const auto [r, s] = filter_shape.dims();
    const auto [n, h, w, c] = x->shape().dims();
    auto kernel =
        b.template covar<R>("kernel", ttl::make_shape(r, s, c, n_filters),
                            nn::ops::truncated_normal<R>(.1));
    auto bias = b.template covar<R>("bias", ttl::make_shape(n_filters),
                                    nn::ops::constant<R>(0.1));

    auto y = b.template invoke<R>("conv", nn::ops::conv<>(), x, kernel);
    auto z = b.template invoke<R>("conv_bias",
                                  nn::ops::add_bias<nn::ops::nhwc>(), y, bias);

    return std::make_tuple(z, kernel, bias);
}
}  // namespace nn::graph::layers
