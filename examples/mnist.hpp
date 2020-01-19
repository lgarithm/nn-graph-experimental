#pragma once
#include <ttl/experimental/copy>
#include <ttl/experimental/get>
#include <ttl/nn/ops>
#include <ttl/range>
#include <ttl/tensor>

#include "trace.hpp"
#include "trainer.hpp"
#include "utils.hpp"

ttl::tensor<float, 2> prepro(const ttl::tensor_view<uint8_t, 1> &t)
{
    const int k = 10;
    ttl::tensor<float, 2> y(t.shape().size(), k);
    (ttl::nn::ops::onehot(k))(ref(y), t);
    return y;
}

// images -> [batch, h * w]
ttl::tensor<float, 2> prepro2(const ttl::tensor_view<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.shape().dims();
    ttl::tensor<float, 2> y(n, h * w);
    std::transform(t.data(), t.data_end(), y.data(),
                   [](uint8_t p) -> float { return p / 255.0; });
    return y;
}

// images -> [batch, h, w, 1]
ttl::tensor<float, 4> prepro4(const ttl::tensor_view<uint8_t, 3> &t)
{
    const auto [n, h, w] = t.shape().dims();
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
               const ttl::nn::graph::var_node *accuracy)
{
    TRACE_SCOPE(__func__);
    std::vector<float> accs;
    simple_data_iterator run_epoch(batch_size);
    run_epoch(images, labels, [&](int /* idx */, auto xs_data, auto y_s_data) {
        TRACE_SCOPE("test batch");
        rt.bind(xs, xs_data);
        rt.bind(y_s, y_s_data);
        b.run(rt, accuracy);
        auto result = accuracy->as<float, 0>()->get_view(rt);
        accs.push_back(ttl::get(result));
    });
    return ttl::mean(ttl::tensor_view<float, 1>(accs.data(), accs.size()));
}

template <typename Images, typename Labels, typename builder, typename RT>
void train_mnist(int epoches, int batch_size,                           //
                 const builder &b, RT &rt,                              //
                 const Images &images, const Labels &labels,            //
                 const Images &test_images, const Labels &test_labels,  //
                 const ttl::nn::graph::var_node *xs,
                 const ttl::nn::graph::var_node *y_s,
                 const ttl::nn::graph::op_node *train_step,
                 const ttl::nn::graph::var_node *accuracy, bool do_test = true)
{
    TRACE_SCOPE(__func__);
    TRACE_STMT(rt.debug());

    simple_trainer run_train(epoches, batch_size, do_test);
    run_train(
        images, labels,
        [&](int idx, auto xs_data, auto y_s_data) {
            TRACE_SCOPE("train batch");
            rt.bind(xs, xs_data);
            rt.bind(y_s, y_s_data);
            b.run(rt, train_step);
        },
        [&](int epoch, int step) {
            const auto acc = test_all(b, rt, batch_size, test_images,
                                      test_labels, xs, y_s, accuracy);
            show_accuracy(acc, epoch + 1, step + 1);
        });
}
