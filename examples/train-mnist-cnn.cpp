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
auto cnn(builder &b, const internal::var_node<R, 4> *x, const shape<2> &ksize,
         int n_filters)
{
    conv_layer<> l(ksize, n_filters);
    ops::truncated_normal kernel_init(0.1);
    ops::constant<R> bias_init(0);
    // ops::readtar kernel_init("cnn-init.tidx", "conv2d/kernel:0");
    // ops::readtar bias_init("cnn-init.tidx", "conv2d/bias:0");
    return l.apply<R>(b, x, kernel_init, bias_init);
}

template <typename R, typename builder>
auto dense(builder &b, const internal::var_node<R, 2> *x, int logits)
{
    dense_layer l(logits);
    ops::truncated_normal weight_init(0.1);
    ops::constant<R> bias_init(0);
    // ops::readtar weight_init("cnn-init.tidx", "dense/kernel:0");
    // ops::readtar bias_init("cnn-init.tidx", "dense/bias:0");
    return l.apply<R>(b, x, weight_init, bias_init);
}
}  // namespace ttl::nn::graph::layers

template <typename R, typename builder>
auto create_cnn_model(builder &b, const ttl::shape<3> &image_shape,
                      int batch_size, int logits)
{
    TRACE_SCOPE(__func__);

    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::cnn;
    using ttl::nn::graph::layers::dense;

    const auto [height, width, channel] = image_shape.dims();
    auto images = b.template var<R>(
        "images", b.shape(batch_size, height, width, channel));
    auto labels = b.template var<uint8_t>("onehot", b.shape(batch_size));
    auto l1 = cnn(b, images, b.shape(3, 3), 32);
    auto l2 = b.template invoke<R>("conv_act", ttl::nn::ops::relu(), *l1);

    auto cnn_flat = b.template invoke<R>(
        "cnn_flat", ttl::nn::ops::copy_flatten<1, 3>(), l2);
    auto l_out = dense(b, cnn_flat, logits);
    auto [predictions, loss, accuracy] =
        classification_output<uint8_t>(b, *l_out, labels, logits);
    return std::make_tuple(images, labels, loss, accuracy);
}

void cnn_cpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_cnn_model<float>(b, b.shape(28, 28, 1), batch_size, 10);

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
                ttl::ref(prepro4(train.images)), ttl::ref(train.labels),  //
                ttl::ref(prepro4(test.images)), ttl::ref(test.labels),    //
                xs, y_s, gvs, accuracy, do_test);
}

template <typename T>
auto make_cuda_tensor_from(const T &t)
{
    ttl::cuda_tensor<typename T::value_type, T::rank> c(t.shape());
    ttl::copy(ref(c), t);
    return c;
}

void cnn_gpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::gpu_builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_cnn_model<float>(b, b.shape(28, 28, 1), batch_size, 10);

    auto gvs = b.gradients(loss);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    auto images = make_cuda_tensor_from(ttl::view(prepro4(train.images)));
    auto labels = make_cuda_tensor_from(ttl::view(train.labels));
    auto test_images = make_cuda_tensor_from(ttl::view(prepro4(test.images)));
    auto test_labels = make_cuda_tensor_from(ttl::view(test.labels));

    train_mnist(epoches, batch_size, b, rt,                    //
                ttl::ref(images), ttl::ref(labels),            //
                ttl::ref(test_images), ttl::ref(test_labels),  //
                xs, y_s, gvs, accuracy, do_test);
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(__func__);
    const int batch_size = 100;
    const int epoches = 3;
    const bool do_test = false;
    if (argc > 1 && std::string(argv[1]) == "gpu") {
        cnn_gpu(batch_size, epoches, do_test);
    } else {
        cnn_cpu(batch_size, epoches, do_test);
    }
    return 0;
}
