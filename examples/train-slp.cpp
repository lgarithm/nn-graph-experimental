#include <experimental/zip>
#include <nn/experimental/bits/ops/utility.hpp>
#include <nn/experimental/datasets>
#include <nn/graph>
#include <nn/ops>
#include <ttl/algorithm>
#include <ttl/tensor>

#include "mnist.hpp"
#include "trace.hpp"
#include "utils.hpp"
#include <nn/bits/ops/chunk.hpp>
#include <nn/contrib/graph/layers/dense.hpp>
#include <nn/contrib/graph/layers/output.hpp>

auto slp(nn::graph::builder &b, int input_size, int batch_size, int logits)
{
    TRACE_SCOPE(__func__);
    using nn::graph::layers::classification_output;
    using nn::graph::layers::dense;
    auto images = b.var<float>("images", b.shape(batch_size, input_size));
    auto labels = b.var<float>("onehot-labels", b.shape(batch_size, logits));
    auto [l1, w1, b1] = dense(b, images, logits);
    auto [loss, accuracy] = classification_output(b, l1, labels);
    return std::make_tuple(images, labels, loss, accuracy);
}

void slp_example(int batch_size, int epoches)
{
    TRACE_SCOPE(__func__);
    nn::graph::builder b;
    const auto [xs, y_s, loss, accuracy] = slp(b, 28 * 28, batch_size, 10);

    nn::graph::optimizer opt;
    auto f = opt.minimize(b, loss, 0.5);

    b.debug();

    nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    auto images = prepro2(train.images);
    auto labels = prepro(train.labels);
    auto test_images = prepro2(test.images);
    auto test_labels = prepro(test.labels);

    const auto image_batches = ttl::chunk(images, batch_size);
    const auto label_batches = ttl::chunk(labels, batch_size);
    std::cerr << "batch size :: " << batch_size << std::endl;
    using std::experimental::zip;
    for (auto epoch : ttl::range(epoches)) {
        TRACE_SCOPE("train epoch");
        for (const auto idx : ttl::range<0>(image_batches)) {
            {
                TRACE_SCOPE("train");
                rt.bind(xs, image_batches[idx]);
                rt.bind(y_s, label_batches[idx]);
                b.run(rt, f);
            }
            bool do_test = true;
            if (do_test) {
                TRACE_SCOPE("test");
                const auto acc = test_all(b, rt, batch_size, test_images,
                                          test_labels, xs, y_s, accuracy);
                show_accuracy(acc, epoch + 1, idx + 1);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(argv[0]);
    const int batch_size = 10000;
    const int epoches = 10;
    slp_example(batch_size, epoches);
    return 0;
}
