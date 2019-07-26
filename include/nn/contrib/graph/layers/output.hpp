#include <tuple>

#include <nn/ops>

#include <nn/bits/graph/builder.hpp>

namespace nn::graph::layers
{
template <typename R, typename builder>
auto classification_output(builder &b,
                           const nn::graph::internal::var_node<R, 2> *out,
                           const nn::graph::internal::var_node<R, 2> *labels)
{
    auto probs = b.template invoke<float>("probs", nn::ops::softmax(), out);
    auto loss =
        b.template invoke<float>("loss", nn::ops::xentropy(), labels, probs);

    auto predictions = b.template invoke<int32_t>(
        "predictions", nn::experimental::ops::argmax(), probs);

    auto accuracy = b.template invoke<float>(
        "accuracy", nn::experimental::ops::similarity(), predictions,
        b.template invoke<int32_t>("labels", nn::experimental::ops::argmax(),
                                   labels));
    return std::make_tuple(loss, accuracy);
}
}  // namespace nn::graph::layers
