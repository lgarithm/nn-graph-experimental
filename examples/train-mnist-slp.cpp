#include <nn/graph>
#include <ttl/algorithm>
#include <ttl/copy>
#include <ttl/nn/experimental/datasets>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "mnist.hpp"
#include "trace.hpp"
#include "trainer.hpp"
#include "utils.hpp"
#include <nn/contrib/graph/layers/dense.hpp>
#include <nn/contrib/graph/layers/output.hpp>

template <typename builder>
auto create_slp_model(builder &b, int input_size, int batch_size, int logits)
{
    TRACE_SCOPE(__func__);
    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::dense;
    auto images =
        b.template var<float>("images", b.shape(batch_size, input_size));
    auto labels =
        b.template var<float>("onehot-labels", b.shape(batch_size, logits));
    auto [l1, w1, b1] = dense(b, images, logits);
    auto [loss, accuracy] = classification_output(b, l1, labels);
    return std::make_tuple(images, labels, loss, accuracy);
}

void slp_cpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] =
        create_slp_model(b, 28 * 28, batch_size, 10);

    ttl::nn::graph::optimizer opt;
    auto f = opt.minimize(b, loss, 0.5);

    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    const auto [images, labels, test_images, test_labels] = [&] {
        TRACE_SCOPE("prepro");
        using nn::experimental::datasets::load_mnist_data;
        const std::string home(std::getenv("HOME"));
        const std::string prefix = home + "/var/data/mnist";
        const auto train = load_mnist_data(prefix, "train");
        const auto test = load_mnist_data(prefix, "t10k");

        auto images = prepro2(train.images);
        auto labels = prepro(train.labels);
        auto test_images = prepro2(test.images);
        auto test_labels = prepro(test.labels);
        return std::make_tuple(std::move(images), std::move(labels),
                               std::move(test_images), std::move(test_labels));
    }();

    train_mnist(epoches, batch_size, b, rt, images, labels, test_images,
                test_labels, xs, y_s, f, accuracy);
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
        create_slp_model(b, 28 * 28, batch_size, 10);

    ttl::nn::graph::optimizer opt;
    auto f = opt.minimize(b, loss, 0.5);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    const auto [images, labels, test_images, test_labels] = [&] {
        TRACE_SCOPE("prepro");
        using nn::experimental::datasets::load_mnist_data;
        const std::string home(std::getenv("HOME"));
        const std::string prefix = home + "/var/data/mnist";
        const auto train = load_mnist_data(prefix, "train");
        const auto test = load_mnist_data(prefix, "t10k");

        auto images_cpu = prepro2(train.images);
        auto labels_cpu = prepro(train.labels);
        auto test_images_cpu = prepro2(test.images);
        auto test_labels_cpu = prepro(test.labels);

        auto images = make_cuda_tensor_from(view(images_cpu));
        auto labels = make_cuda_tensor_from(view(labels_cpu));
        auto test_images = make_cuda_tensor_from(view(test_images_cpu));
        auto test_labels = make_cuda_tensor_from(view(test_labels_cpu));
        return std::make_tuple(std::move(images), std::move(labels),
                               std::move(test_images), std::move(test_labels));
    }();

    train_mnist(epoches, batch_size, b, rt, images, labels, test_images,
                test_labels, xs, y_s, f, accuracy, do_test);
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
