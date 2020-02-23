#include <experimental/zip>
#include <nn/graph>
#include <stdml/control>
#include <ttl/nn/experimental/datasets>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "mnist.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include <ttl/nn/contrib/graph/layers/conv.hpp>
#include <ttl/nn/contrib/graph/layers/dense.hpp>
#include <ttl/nn/contrib/graph/layers/output.hpp>

DEFINE_TRACE_CONTEXTS;

template <typename builder>
auto create_cnn_model(builder &b, const ttl::shape<3> &image_shape,
                      int batch_size, int logits)
{
    TRACE_SCOPE(__func__);

    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::cnn;
    using ttl::nn::graph::layers::dense;

    const auto [height, width, channel] = image_shape.dims();
    auto images = b.template var<float>(
        "images", b.shape(batch_size, height, width, channel));
    auto labels =
        b.template var<float>("onehot-labels", b.shape(batch_size, logits));
    auto [l1, w1, b1] = cnn(b, images, b.shape(3, 3), 32);
    auto l2 = b.template invoke<float>("conv_act", ttl::nn::ops::relu(), l1);
    // auto [l2, w2, b2] = cnn(b, l1, b.shape(3, 3), 32);

    auto cnn_flat = b.template invoke<float>(
        "cnn_flat", ttl::nn::ops::copy_flatten<1, 3>(), l2);

    auto [l_out, w3, b3] = dense(b, cnn_flat, logits);
    auto [loss, accuracy] = classification_output(b, l_out, labels);

    return std::make_tuple(images, labels, loss, accuracy);
}

void cnn_cpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_cnn_model(b, b.shape(28, 28, 1), batch_size, 10);

    auto gvs = b.gradients(loss);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    auto images = prepro4(train.images);
    auto labels = prepro(train.labels);
    auto test_images = prepro4(test.images);
    auto test_labels = prepro(test.labels);

    train_mnist(epoches, batch_size, b, rt, images, labels, test_images,
                test_labels, xs, y_s, gvs, accuracy, do_test);
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
        create_cnn_model(b, b.shape(28, 28, 1), batch_size, 10);

    auto gvs = b.gradients(loss);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    auto images_cpu = prepro4(train.images);
    auto labels_cpu = prepro(train.labels);
    auto test_images_cpu = prepro4(test.images);
    auto test_labels_cpu = prepro(test.labels);

    auto images = make_cuda_tensor_from(ttl::view(images_cpu));
    auto labels = make_cuda_tensor_from(ttl::view(labels_cpu));
    auto test_images = make_cuda_tensor_from(ttl::view(test_images_cpu));
    auto test_labels = make_cuda_tensor_from(ttl::view(test_labels_cpu));

    train_mnist(epoches, batch_size, b, rt, images, labels, test_images,
                test_labels, xs, y_s, gvs, accuracy, do_test);
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
