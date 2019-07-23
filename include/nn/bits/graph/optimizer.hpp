#pragma once
#include <nn/bits/graph/builder.hpp>
#include <nn/bits/graph/runtime.hpp>

namespace nn::graph::internal
{
template <typename D> struct get_apply_op;

using var_node_list_t = std::vector<const base_var_node *>;

template <> struct get_apply_op<cpu> {
    auto operator()(const base_var_node *v, float eta,
                    const var_node_list_t &gs) const
    {
        return v->apply_gradients_cpu(eta, gs);
    }
};

template <> struct get_apply_op<nvidia_gpu> {
    auto operator()(const base_var_node *v, float eta,
                    const var_node_list_t &gs) const
    {
        return v->apply_gradients_gpu(eta, gs);
    }
};

class optimizer
{
    using grad_var_t = std::pair<const base_var_node *, const base_var_node *>;
    using var_node_list_t = std::vector<const base_var_node *>;
    using var_grads_t = std::map<const base_var_node *, var_node_list_t>;

    static auto group_gradients(const std::vector<grad_var_t> &gvs)
    {
        var_grads_t var_grads;
        for (auto [g, v] : gvs) { var_grads[v].push_back(g); }
        return var_grads;
    }

  public:
    template <typename D, typename R, ttl::rank_t r>
    op_node *minimize(base_builder &b, const var_node<R, r> *l,
                      const float &eta = 0.1) const
    {
        const auto gvs = b.gradients<R, r, D>(l);
        const auto var_grads = group_gradients(gvs);

        std::vector<const node *> apply_ops;
        for (auto [v, gs] : var_grads) {
            const auto op = get_apply_op<D>()(v, eta, gs);
            apply_ops.push_back(op);
            b.own(op);
        }

        using RT = basic_runtime<D>;
        return b.op("minimize", [](RT &) {}, apply_ops);
    }

    template <typename R, ttl::rank_t r>
    op_node *minimize(base_builder &b, const var_node<R, r> *l,
                      const float &eta = 0.1) const
    {
        return minimize<cpu>(b, l, eta);
    }
};
}  // namespace nn::graph::internal
