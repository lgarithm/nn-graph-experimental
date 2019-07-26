#include <experimental/zip>
#include <nn/experimental/bits/ops/utility.hpp>
#include <nn/experimental/datasets>
#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

#include "mnist.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include <nn/bits/ops/chunk.hpp>
#include <nn/bits/ops/constant.hpp>
#include <nn/contrib/graph/layers/conv.hpp>
#include <nn/contrib/graph/layers/dense.hpp>
#include <nn/contrib/graph/layers/output.hpp>

auto cnn_model(nn::graph::builder &b, const ttl::shape<3> &image_shape,
               int batch_size, int logits)
{
    TRACE_SCOPE(__func__);

    using nn::graph::layers::classification_output;
    using nn::graph::layers::cnn;
    using nn::graph::layers::dense;

    const auto [height, width, channel] = image_shape.dims();
    auto images =
        b.var<float>("images", b.shape(batch_size, height, width, channel));
    auto labels = b.var<float>("onehot-labels", b.shape(batch_size, logits));
    auto [l1, w1, b1] = cnn(b, images, b.shape(3, 3), 32);
    auto [l2, w2, b2] = cnn(b, l1, b.shape(3, 3), 32);

    auto cnn_flat = b.invoke<float>(
        "cnn_flat",
        nn::ops::reshape_copy<2>(nn::ops::as_mat_shape<1, 3>(l2->shape())), l2);

    auto [l_out, w3, b3] = dense(b, cnn_flat, logits);
    auto [loss, accuracy] = classification_output(b, l_out, labels);

    return std::make_tuple(images, labels, loss, accuracy);
}

void train_cnn(int batch_size, int epoches)
{
    TRACE_SCOPE(__func__);
    nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] =
        cnn_model(b, b.shape(28, 28, 1), batch_size, 10);

    nn::graph::internal::optimizer opt;
    auto f = opt.minimize(b, loss, 0.01);

    nn::graph::runtime rt;
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

    train_mnist("cpu", epoches, batch_size, b, rt, images, labels, test_images,
                test_labels, xs, y_s, f, accuracy);
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(argv[0]);
    const int batch_size = 50;
    const int epoches = 10;
    train_cnn(batch_size, epoches);
    return 0;
}
