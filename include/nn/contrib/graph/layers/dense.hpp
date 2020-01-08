
#include <tuple>

#include <ttl/nn/ops>

#include <nn/bits/graph/builder.hpp>

namespace ttl::nn::graph::layers
{
template <typename R, typename builder>
auto dense(builder &b, const nn::graph::internal::var_node<R, 2> *t, int logits)
{
    auto weight = b.template covar<R>(
        "w", ttl::make_shape(std::get<1>(t->shape().dims()), logits),
        nn::ops::constant<R>(.5));
    auto bias = b.template covar<R>("b", ttl::make_shape(logits),
                                    nn::ops::constant<R>(0.0));
    auto l = b.template invoke<R>(
        "prod", nn::ops::add_bias<nn::ops::hw>(),
        b.template invoke<R>("biased", nn::ops::matmul(), t, weight), bias);
    return std::make_tuple(l, weight, bias);
}
}  // namespace ttl::nn::graph::layers
