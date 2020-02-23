
#include <stdml/experimental/models>
#include <ttl/algorithm>
#include <ttl/nn/computation_graph>
#include <ttl/nn/contrib/graph/layers/output.hpp>
#include <ttl/nn/experimental/datasets>
#include <ttl/nn/graph/layers>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "trace.hpp"

DEFINE_TRACE_CONTEXTS;

namespace ttl::nn::graph::layers
{
template <typename R, typename builder>
auto dense(builder &b, const internal::var_node<R, 2> *x, int logits)
{
    dense_layer l(logits);
    ops::constant<R> weight_init(0.5);
    ops::zeros bias_init;
    return l.apply<R>(b, x, weight_init, bias_init);
}
}  // namespace ttl::nn::graph::layers

// images -> [batch, h * w]
ttl::tensor<float, 2> prepro2(const ttl::tensor_view<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.dims();
    ttl::tensor<float, 2> y(n, h * w);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

template <typename R, typename builder>
auto create_slp_model(builder &b, int input_size, int batch_size, int logits)
{
    TRACE_SCOPE(__func__);
    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::dense;
    auto sample = b.template var<R>("sample", b.shape(batch_size, input_size));
    auto labels = b.template var<uint8_t>("labels", b.shape(batch_size));
    auto l1 = dense(b, sample, logits);
    auto [predictions, loss, accuracy] =
        classification_output<uint8_t, R>(b, *l1, labels, logits);
    return std::make_tuple(sample, labels, predictions, loss, accuracy);
}

void slp_model()
{
    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    const int batch_size = 100;
    const int epochs = 1;
    stdml::classification_model<1, uint8_t> model(ttl::make_shape(28 * 28), 10);

    model.init(
        [](auto &b, const ttl::shape<1> &in, const int k, const int bs) {
            return create_slp_model<float>(b, in.size(), bs, k);
        },
        batch_size);

    model.train(ttl::view(prepro2(train.images)), ttl::view(train.labels),
                batch_size, epochs);
    model.test(ttl::view(prepro2(train.images)), ttl::view(train.labels),
               batch_size);
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    slp_model();
    return 0;
}
