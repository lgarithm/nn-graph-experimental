#pragma once
#include <stdml/experimental/control>
#include <ttl/experimental/copy>
#include <ttl/experimental/get>
#include <ttl/nn/cuda_ops>
#include <ttl/nn/ops>
#include <ttl/range>
#include <ttl/tensor>

#include "utils.hpp"

// images -> [batch, h * w]
ttl::tensor<float, 2> prepro2(const ttl::tensor_view<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.dims();
    ttl::tensor<float, 2> y(n, h * w);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

// images -> [batch, h, w, 1]
ttl::tensor<float, 4> prepro4(const ttl::tensor_view<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.dims();
    ttl::tensor<float, 4> y(n, h, w, 1);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

template <typename Images, typename Labels, typename builder, typename RT>
float test_all(const builder &b, RT &rt,  //
               int batch_size, const Images &images, const Labels &labels,
               const ttl::nn::graph::var_node *xs,
               const ttl::nn::graph::var_node *y_s,
               const ttl::nn::graph::var_node *predications)
{
    TRACE_SCOPE(__func__);
    std::vector<float> accs;
    stdml::batch_invoke(
        batch_size,
        [&](auto xs_data, auto y_s_data) {
            TRACE_SCOPE("test batch");
            rt.bind(xs, xs_data);
            rt.bind(y_s, y_s_data);
            b.run(rt, predications);
            auto result = rt.template view<uint8_t, 1>(predications);
            ttl::tensor<float, 0, typename Images::device_type> acc;
            ttl::nn::ops::similarity()(ttl::ref(acc), y_s_data, result);
            accs.push_back(ttl::get(ttl::view(acc)));
        },
        images, labels);
    return ttl::mean(ttl::tensor_view<float, 1>(accs.data(), accs.size()));
}

template <typename Images, typename Labels, typename builder, typename RT>
void train_mnist(int epoches, int batch_size,                           //
                 const builder &b, RT &rt,                              //
                 const Images &images, const Labels &labels,            //
                 const Images &test_images, const Labels &test_labels,  //
                 const ttl::nn::graph::var_node *xs,
                 const ttl::nn::graph::var_node *y_s,
                 const std::vector<std::pair<const ttl::nn::graph::var_node *,
                                             const ttl::nn::graph::var_node *>>
                     gvs,
                 const ttl::nn::graph::var_node *predications,
                 bool do_test = true)
{
    TRACE_SCOPE(__func__);
    TRACE_STMT(rt.debug());
    const auto gs = firsts(gvs);
    const float lr = 0.1;
    for (auto e[[gnu::unused]] : ttl::range(epoches)) {
        stdml::batch_invoke(batch_size,
                            [&](auto xs_data, auto y_s_data) {
                                rt.bind(xs, xs_data);
                                rt.bind(y_s, y_s_data);
                                b.run(rt, gs);
                                stdml::internal::learn_all<float>(gvs, rt, lr);
                            },
                            images, labels);
    }
    printf("train finished\n");
    const auto acc = test_all(b, rt, batch_size, test_images, test_labels, xs,
                              y_s, predications);
    show_accuracy(acc, epoches, 0);
    printf("test finished\n");
}
