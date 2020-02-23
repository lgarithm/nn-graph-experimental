#include <ttl/algorithm>
#include <ttl/experimental/copy>
#include <ttl/nn/computation_graph>
#include <ttl/nn/contrib/graph/layers/output.hpp>
#include <ttl/nn/experimental/datasets>
#include <ttl/nn/graph/layers>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "mnist.hpp"
#include "utils.hpp"

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

template <typename R, typename builder>
auto create_slp_model(builder &b, int input_size, int batch_size, int logits)
{
    TRACE_SCOPE(__func__);
    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::dense;
    auto images = b.template var<R>("images", b.shape(batch_size, input_size));
    auto labels = b.template var<uint8_t>("labels", b.shape(batch_size));
    auto l1 = dense(b, images, logits);
    auto [predictions, loss, accuracy] =
        classification_output<uint8_t>(b, *l1, labels, logits);
    return std::make_tuple(images, labels, loss, accuracy);
}

void slp_cpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_slp_model<float>(b, 28 * 28, batch_size, 10);

    auto gvs = b.gradients(loss);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    train_mnist(epoches, batch_size, b, rt,                               //
                ttl::ref(prepro2(train.images)), ttl::ref(train.labels),  //
                ttl::ref(prepro2(test.images)), ttl::ref(test.labels),    //
                xs, y_s, gvs, accuracy);
}

template <typename T>
auto make_cuda_tensor_from(const T &t)
{
    ttl::cuda_tensor<typename T::value_type, T::rank> c(t.shape());
    ttl::copy(ref(c), t);
    return c;
}

void slp_gpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::gpu_builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_slp_model<float>(b, 28 * 28, batch_size, 10);

    auto gvs = b.gradients(loss);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    auto images = make_cuda_tensor_from(ttl::view(prepro2(train.images)));
    auto labels = make_cuda_tensor_from(ttl::view(train.labels));
    auto test_images = make_cuda_tensor_from(ttl::view(prepro2(test.images)));
    auto test_labels = make_cuda_tensor_from(ttl::view(test.labels));

    train_mnist(epoches, batch_size, b, rt,                    //
                ttl::ref(images), ttl::ref(labels),            //
                ttl::ref(test_images), ttl::ref(test_labels),  //
                xs, y_s, gvs, accuracy);
}

void show_args(int argc, char *argv[])
{
    printf("argc=%d\n", argc);
    for (int i = 0; i < argc; ++i) { printf("argv[%d]=%s\n", i, argv[i]); }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    show_args(argc, argv);
    const int batch_size = 10000;
    const int epoches = 10;
    const bool do_test = true;
    if (argc > 1 && std::string(argv[1]) == "gpu") {
        slp_gpu(batch_size, epoches, do_test);
    } else {
        slp_cpu(batch_size, epoches, do_test);
    }
    return 0;
}
