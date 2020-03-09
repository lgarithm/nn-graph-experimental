#include <stdml/ds/mnist>
#include <stdml/prepro>
#include <ttl/algorithm>
#include <ttl/experimental/copy>
#include <ttl/nn/computation_graph>
#include <ttl/nn/graph/layers>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "mnist.hpp"
#include "utils.hpp"

DEFINE_TRACE_CONTEXTS;

namespace ttl::nn::graph::layers
{
template <typename R, typename builder>
auto dense(builder &b, const internal::var_node<R, 2> *x, int logits)
{
    dense_layer l(logits);
    ops::constant<R> weight_init(0.5);
    ops::zeros bias_init;
    return l.apply<R>(b, x, weight_init, bias_init);
}
}  // namespace ttl::nn::graph::layers

template <typename R, typename builder>
auto create_slp_model(builder &b, int input_size, int batch_size, int logits)
{
    TRACE_SCOPE(__func__);
    using ttl::nn::graph::layers::classification_output;
    using ttl::nn::graph::layers::dense;
    auto images = b.template var<R>("images", b.shape(batch_size, input_size));
    auto labels = b.template var<uint8_t>("labels", b.shape(batch_size));
    auto l1 = dense(b, images, logits);
    auto [predictions, loss] = classification_output()(b, *l1, labels);
    return std::make_tuple(images, labels, predictions, loss);
}

void slp_cpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::builder b;
    const auto [xs, y_s, predictions, loss] =
        create_slp_model<float>(b, 28 * 28, batch_size, 10);
    auto gvs = b.gradients(loss);
    ttl::nn::graph::runtime rt;
    b.build(rt);
    b.init(rt);
    using mnist = stdml::datasets::mnist<>;
    const auto ds = mnist::load_all();
    stdml::experimental::prepro_slp prepro;
    train_mnist(
        epoches, batch_size, b, rt,                                      //
        ttl::view(prepro(ds.train.images)), ttl::view(ds.train.labels),  //
        ttl::view(prepro(ds.test.images)), ttl::view(ds.test.labels),    //
        xs, y_s, gvs, predictions);
}

void slp_gpu(int batch_size, int epoches, bool do_test)
{
    TRACE_SCOPE(__func__);
    ttl::nn::graph::gpu_builder b;
    const auto [xs, y_s, predictions, loss] =
        create_slp_model<float>(b, 28 * 28, batch_size, 10);
    auto gvs = b.gradients(loss);
    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);
    using mnist = stdml::datasets::mnist<ttl::cuda_memory>;
    const auto ds = mnist::load_all();
    stdml::experimental::prepro_slp prepro;
    train_mnist(
        epoches, batch_size, b, rt,                                      //
        ttl::view(prepro(ds.train.images)), ttl::view(ds.train.labels),  //
        ttl::view(prepro(ds.test.images)), ttl::view(ds.test.labels),    //
        xs, y_s, gvs, predictions);
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
