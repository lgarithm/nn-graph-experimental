
#include <tuple>

#include <nn/ops>

#include <nn/bits/graph/builder.hpp>

namespace nn::graph::layers
{
template <typename R>
auto dense(nn::graph::builder &b, const nn::graph::internal::var_node<R, 2> *t,
           int logits)
{
    auto weight =
        b.covar<R>("w", ttl::make_shape(std::get<1>(t->shape().dims()), logits),
                   nn::ops::constant<R>(.5));
    auto bias =
        b.covar<R>("b", ttl::make_shape(logits), nn::ops::constant<R>(0.0));
    auto l =
        b.invoke<R>("prod", nn::ops::add_bias<nn::ops::hw>(),
                    b.invoke<R>("biased", nn::ops::matmul(), t, weight), bias);
    return std::make_tuple(l, weight, bias);
}
}  // namespace nn::graph::layers
