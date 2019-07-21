#pragma once
#include <nn/ops>
#include <ttl/range>
#include <ttl/tensor>

#include "trace.hpp"

ttl::tensor<float, 2> prepro(const ttl::tensor<uint8_t, 1> &t)
{
    const int k = 10;
    ttl::tensor<float, 2> y(t.shape().size(), k);
    (nn::experimental::ops::onehot(k))(ref(y), view(t));
    return y;
}

// images -> [batch, h * w]
ttl::tensor<float, 2> prepro2(const ttl::tensor<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.shape().dims();
    ttl::tensor<float, 2> y(n, h * w);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

// images -> [batch, h, w, 1]
ttl::tensor<float, 4> prepro4(const ttl::tensor<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.shape().dims();
    ttl::tensor<float, 4> y(n, h, w, 1);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

template <typename Images, typename Labels>
float test_all_clang(const nn::graph::builder &b, nn::graph::runtime &rt,
                     int batch_size, const Images &images, const Labels &labels,
                     const nn::graph::var_node *xs,
                     const nn::graph::var_node *y_s,
                     const nn::graph::var_node *accuracy)
{
    const auto image_batches = chunk(images, batch_size);
    const auto label_bathces = chunk(labels, batch_size);
    const auto n = std::get<0>(image_batches.shape().dims());
    ttl::tensor<float, 1> accs(n);
    for (const auto [i, images, labels] :
         zip(ttl::range(n), image_batches, label_bathces)) {
        LOG_SCOPE("train batch");
        std::cerr << "test batch " << i + 1 << "/" << n << std::endl;
        rt.bind(xs, images);
        rt.bind(y_s, labels);
        b.run(rt, accuracy);
        accs[i] = accuracy->as<float, 0>()->get_view(rt).data()[0];
    }
    return ttl::mean(view(accs));
}

template <typename Images, typename Labels>
float test_all_gcc(const nn::graph::builder &b, nn::graph::runtime &rt,
                   int batch_size, const Images &images, const Labels &labels,
                   const nn::graph::var_node *xs,
                   const nn::graph::var_node *y_s,
                   const nn::graph::var_node *accuracy)
{
    const auto image_batches = chunk(images, batch_size);
    const auto label_bathces = chunk(labels, batch_size);
    const auto n = std::get<0>(image_batches.shape().dims());
    ttl::tensor<float, 1> accs(n);
    for (const auto i : ttl::range(n)) {
        // LOG_SCOPE("test batch");
        // std::cerr << "test batch " << i + 1 << "/" << n << std::endl;
        rt.bind(xs, image_batches[i]);
        rt.bind(y_s, label_bathces[i]);
        b.run(rt, accuracy);
        accs[i] = accuracy->as<float, 0>()->get_view(rt).data()[0];
    }
    return ttl::mean(view(accs));
}

template <typename Images, typename Labels>
float test_all(const nn::graph::builder &b, nn::graph::runtime &rt,
               int batch_size, const Images &images, const Labels &labels,
               const nn::graph::var_node *xs, const nn::graph::var_node *y_s,
               const nn::graph::var_node *accuracy)
{
    return test_all_gcc(b, rt, batch_size, images, labels, xs, y_s, accuracy);
    // FIXME: make it work for gcc
    //  test_all_clang(b, rt, batch_size, images, labels, xs, y_s, accuracy);
}
