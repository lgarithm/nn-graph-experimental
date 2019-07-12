#include <tuple>

#include <nn/ops>

#include <nn/bits/graph/builder.hpp>

namespace nn::graph::layers
{
template <typename R>
auto classification_output(nn::graph::builder &b,
                           const nn::graph::internal::var_node<R, 2> *out,
                           const nn::graph::internal::var_node<R, 2> *labels)
{
    auto probs = b.invoke<float>("probs", nn::ops::softmax(), out);
    auto loss = b.invoke<float>("loss", nn::ops::xentropy(), labels, probs);

    auto predictions = b.invoke<int32_t>(
        "predictions", nn::experimental::ops::argmax(), probs);

    auto accuracy = b.invoke<float>(
        "accuracy", nn::experimental::ops::similarity(), predictions,
        b.invoke<int32_t>("labels", nn::experimental::ops::argmax(), labels));
    return std::make_tuple(loss, accuracy);
}
}  // namespace nn::graph::layers
