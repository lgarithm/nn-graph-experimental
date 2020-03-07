#include <iostream>

#include <stdml/ds/mnist>
#include <ttl/debug>
#include <ttl/experimental/zip>
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/range>

int main(int argc, char *argv[])
{
    // TRACE_SCOPE(argv[0]);

    const auto ds = stdml::datasets::mnist::load_all();

    std::cerr << ttl::type_of(ds.train.images).name() << std::endl;
    std::cerr << ttl::type_of(ds.train.labels).name() << std::endl;
    std::cerr << ttl::type_of(ds.test.images).name() << std::endl;
    std::cerr << ttl::type_of(ds.test.labels).name() << std::endl;

    const int N = 5000;
    const auto images = chunk(ds.train.images, N);
    const auto labels = chunk(ds.train.labels, N);
    using ttl::chunk;
    using ttl::range;
    using ttl::experimental::zip;
    {
        int idx = 0;
        for (const auto [images, labels] : zip(images, labels)) {
            std::cerr << ++idx << std::endl;
            std::cerr << ttl::type_of(images).name() << std::endl;
            std::cerr << ttl::type_of(labels).name() << std::endl;
        }
    }

    {
        const auto r = range<0>(images);
        for (const auto [idx, images, labels] : zip(r, images, labels)) {
            std::cerr << idx << std::endl;
            std::cerr << ttl::type_of(images).name() << std::endl;
            std::cerr << ttl::type_of(labels).name() << std::endl;
        }
    }
    return 0;
}
