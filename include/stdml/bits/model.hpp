#pragma once
#include <stdml/bits/control.hpp>
#include <ttl/device>
#include <ttl/nn/computation_graph>
#include <ttl/shape>
#include <ttl/tensor>

namespace stdml::internal
{
namespace
{
template <typename R, ttl::rank_t r>
static ttl::tensor_ref<R, r> force_ref(const ttl::tensor_view<R, r> &t)
{
    return ttl::tensor_ref<R, r>(const_cast<R *>(t.data()), t.shape());
}
}  // namespace

class basic_supervised_model
{
  public:
    // virtual void train() = 0;
    // virtual void test() = 0;
};

template <typename ttl::rank_t r, typename N, typename D = ttl::host_memory>
class basic_classification_model  // : public basic_supervised_model
{
    const ttl::shape<r> input;
    const N n_categories;

    ttl::nn::graph::builder b;
    ttl::nn::graph::runtime rt;

    using var_node = ttl::nn::graph::var_node;

    var_node *xs = nullptr;
    var_node *y_s = nullptr;
    var_node *predictions = nullptr;
    var_node *loss = nullptr;
    var_node *accuracy = nullptr;

    using grad_var_t = std::pair<const var_node *, const var_node *>;
    std::vector<grad_var_t> gvs;
    std::vector<const var_node *> train_step_ops;

    template <typename R>
    R train_batch(const ttl::tensor_view<R, r + 1, D> &samples,
                  const ttl::tensor_view<N, 1, D> &labels)
    {
        rt.bind(xs, force_ref(samples));
        rt.bind(y_s, force_ref(labels));
        b.run(rt, train_step_ops);
        ttl::tensor_view<R, 1> loss =
            rt.get_raw_view(this->loss).template typed<R, 1>();
        return ttl::mean(loss);
    }

    template <typename R>
    int test_batch(const ttl::tensor_view<R, r + 1, D> &samples,
                   const ttl::tensor_view<N, 1, D> &labels)
    {
        // FIXME: allow bind tensor_view
        rt.bind(xs, force_ref(samples));
        rt.bind(y_s, force_ref(labels));
        b.run(rt, predictions);
        auto y_out = rt.get_raw_view(predictions).template typed<N, 1>();
        return ttl::hamming_distance(y_out, labels);
    }

    template <typename R, typename U, typename V>
    static R percent(U x, V y)
    {
        return static_cast<R>(100) * static_cast<R>(x) / static_cast<R>(y);
    }

  public:
    basic_classification_model(const ttl::shape<r> &input, const N n_categories)
        : input(input), n_categories(n_categories)
    {
    }

    template <typename F>
    void init(const F &create_model, const int batch_size)
    {
        std::tie(xs, y_s, predictions, loss, accuracy) =
            create_model(b, input, n_categories, batch_size);
        gvs = b.gradients(loss->as<float, 1>());
        for (const auto &[g, v] : gvs) {
            printf("%s is the gradient of %s\n", g->name().c_str(),
                   v->name().c_str());
            train_step_ops.push_back(g);
        }
        // train_step_ops.push_back(accuracy);
        b.build(rt);
        b.init(rt);
    }

    template <typename R>
    void train(const ttl::tensor_view<R, r + 1, D> &samples,
               const ttl::tensor_view<N, 1, D> &labels,  //
               const int batch_size, const int epochs = 1)
    {
        int step = 0;
        for (auto epoch [[gnu::unused]] : ttl::range(epochs))
            batch_invoke(
                batch_size,
                [&](const ttl::tensor_view<R, r + 1, D> &samples,
                    const ttl::tensor_view<N, 1, D> &labels) {
                    ++step;
                    const R loss = train_batch(samples, labels);
                    learn_all<R>(gvs, rt, 0.1);
                    if (step % 100 == 0) {
                        printf("step %4d, loss: %f\n", step, loss);
                    }
                },
                samples, labels);
    }

    template <typename R>
    void test(const ttl::tensor_view<R, r + 1, D> &samples,
              const ttl::tensor_view<N, 1, D> &labels,  //
              const int batch_size)
    {
        int tot_succ = 0;
        int tot_failed = 0;
        batch_invoke(
            batch_size,
            [&](const ttl::tensor_view<R, r + 1, D> &samples,
                const ttl::tensor_view<N, 1, D> &labels) {
                const int failed = test_batch(samples, labels);
                tot_succ += std::get<0>(samples.dims()) - failed;
                tot_failed += failed;
            },
            samples, labels);
        printf("succ: %d, failed: %d, accuracy: %.2f%%\n", tot_succ, tot_failed,
               percent<float>(tot_succ, tot_succ + tot_failed));
    }
};
}  // namespace stdml::internal
