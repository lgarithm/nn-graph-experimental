#pragma once
#include <stdml/experimental/control>
#include <ttl/nn/computation_graph>

namespace stdml::internal
{
// m : T<Rx,rx> -> T<Ry,rx>
template <typename Rx, typename ttl::rank_t rx, typename Ry, ttl::rank_t ry,
          typename D = ttl::host_memory>
class basic_supervised_model
{
  protected:
    const ttl::shape<rx> sx;
    const ttl::shape<ry> sy;

    template <typename R, ttl::rank_t r>
    using Var = ttl::nn::graph::internal::var_node<R, r>;

    using var_node = ttl::nn::graph::var_node;

    Var<Rx, rx + 1> *samples_ = nullptr;
    Var<Ry, ry + 1> *labels_ = nullptr;
    Var<Ry, ry + 1> *outputs_ = nullptr;
    var_node *loss_ = nullptr;

    using grad_var_t = std::pair<const var_node *, const var_node *>;
    std::vector<grad_var_t> gvs;
    std::vector<const var_node *> train_step_ops;

    ttl::nn::graph::builder b;
    ttl::nn::graph::runtime rt;

    using Xs = ttl::tensor_view<Rx, rx + 1, D>;
    using Ys = ttl::tensor_view<Ry, ry + 1, D>;

    template <typename Result>
    void test_batch(Result &result, const Xs &samples, const Ys &labels)
    {
        {
            ttl::nn::graph::internal::binding _1(rt, samples_, samples);
            ttl::nn::graph::internal::binding _2(rt, labels_, labels);
            b.run(rt, outputs_);
        }
        result.add(rt.view(outputs_), labels);
    }

  public:
    using samples_type = Xs;
    using labels_type = Ys;

    basic_supervised_model(const ttl::shape<rx> &sx, const ttl::shape<ry> &sy)
        : sx(sx), sy(sy)
    {
    }

    std::vector<ttl::flat_shape> parameter_types() const
    {
        std::vector<ttl::flat_shape> types;
        for (const auto &[g, v] : gvs) { types.push_back(v->type()); }
        return types;
    }

    const std::vector<grad_var_t> &parameter_gradients() const { return gvs; }

    ttl::nn::graph::runtime &get_runtime() { return rt; }

    Rx train_batch(const Xs &samples, const Ys &labels)
    {
        {
            ttl::nn::graph::internal::binding _1(rt, samples_, samples);
            ttl::nn::graph::internal::binding _2(rt, labels_, labels);
            b.run(rt, train_step_ops);
        }
        return ttl::mean(rt.view<Rx, 1>(loss_));
    }

    // template <typename optimizer>
    void train(/* optimizer &opt, */ const Xs &samples, const Ys &labels,
               const int batch_size, const int epochs = 1)
    {
        for (auto epoch [[gnu::unused]] : ttl::range(epochs))
            batch_invoke(
                batch_size,
                [&](const Xs &samples, const Ys &labels) {
                    // const R loss =
                    train_batch(samples, labels);

                    // after train before learn
                    // opt.template call<Rx>(gvs, rt);
                    learn_all<Rx>(gvs, rt, 0.1);
                },
                samples, labels);
    }

    template <typename Result>
    void test(Result &result, const Xs &samples, const Ys &labels,
              const int batch_size)
    {
        batch_invoke(
            batch_size,
            [&](const Xs &samples, const Ys &labels) {
                test_batch(result, samples, labels);
            },
            samples, labels);
    }
};
}  // namespace stdml::internal
