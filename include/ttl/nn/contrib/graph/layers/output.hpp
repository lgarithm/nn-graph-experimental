#include <tuple>

#include <ttl/nn/bits/graph/builder.hpp>
#include <ttl/nn/ops>

namespace ttl::nn::graph::layers
{
template <typename N, typename R, typename builder>
auto classification_output(builder &b, const internal::var_node<R, 2> *out,
                           const internal::var_node<N, 1> *labels,
                           const int n_categories)
{
    auto predictions = b.template invoke<N>("predictions", ops::argmax(), out);
    auto probs = b.template invoke<R>("probs", ops::softmax(), out);
    auto onehot_labels = b.template invoke<R>(
        "onehot-labels", ops::onehot(n_categories), labels);
    auto loss =
        b.template invoke<R>("loss", ops::xentropy(), onehot_labels, probs);
    auto accuracy = b.template invoke<R>("accuracy", ops::similarity(),
                                         predictions, labels);
    return std::make_tuple(predictions, loss, accuracy);
}
}  // namespace ttl::nn::graph::layers
