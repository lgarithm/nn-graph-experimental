#include <experimental/zip>
#include <iostream>

#include <ttl/debug>
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/nn/experimental/datasets>
#include <ttl/range>

int main(int argc, char *argv[])
{
    // TRACE_SCOPE(argv[0]);

    using nn::experimental::datasets::load_mnist_data;
    const std::string home(std::getenv("HOME"));
    const std::string prefix = home + "/var/data/mnist";
    const auto train = load_mnist_data(prefix, "train");
    const auto test = load_mnist_data(prefix, "t10k");

    std::cerr << ttl::tensor_type_name(train.images) << std::endl;
    std::cerr << ttl::tensor_type_name(train.labels) << std::endl;
    std::cerr << ttl::tensor_type_name(test.images) << std::endl;
    std::cerr << ttl::tensor_type_name(test.labels) << std::endl;

    const int N = 5000;
    const auto images = chunk(train.images, N);
    const auto labels = chunk(train.labels, N);
    using std::experimental::zip;
    using ttl::chunk;
    using ttl::range;

    {
        int idx = 0;
        for (const auto [images, labels] : zip(images, labels)) {
            std::cerr << ++idx << std::endl;
            std::cerr << ttl::tensor_type_name(images) << std::endl;
            std::cerr << ttl::tensor_type_name(labels) << std::endl;
        }
    }

    {
        const auto r = range<0>(images);
        for (const auto [idx, images, labels] : zip(r, images, labels)) {
            std::cerr << idx << std::endl;
            std::cerr << ttl::tensor_type_name(images) << std::endl;
            std::cerr << ttl::tensor_type_name(labels) << std::endl;
        }
    }
    return 0;
}
