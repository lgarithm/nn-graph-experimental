#pragma once
#include <tuple>

#include <ttl/nn/bits/graph/builder.hpp>
#include <ttl/nn/ops>

namespace ttl::nn::graph::layers
{
class classification_output
{
  public:
    template <typename N, typename R, typename builder>
    auto operator()(builder &b, const internal::var_node<R, 2> *out,
                    const internal::var_node<N, 1> *labels)
    {
        const auto n_categories = std::get<1>(out->shape().dims());
        auto preds = b.template invoke<N>("preds", ops::argmax(), out);
        auto probs = b.template invoke<R>("probs", ops::softmax(), out);
        auto onehot_labels = b.template invoke<R>(
            "onehot-labels", ops::onehot(n_categories), labels);
        auto loss =
            b.template invoke<R>("loss", ops::xentropy(), onehot_labels, probs);
        return std::make_tuple(preds, loss);
    }
};
}  // namespace ttl::nn::graph::layers
