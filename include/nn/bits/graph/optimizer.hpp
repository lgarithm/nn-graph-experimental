#pragma once
#include <nn/bits/graph/builder.hpp>
#include <nn/bits/graph/runtime.hpp>

namespace nn::graph::internal
{

using var_node_list_t = std::vector<const base_var_node *>;

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
    template <typename R, ttl::rank_t r>
    op_node *minimize(base_builder &b, const var_node<R, r> *l,
                      const float &eta = 0.1) const
    {
        const auto gvs = b.gradients(l);
        const auto var_grads = group_gradients(gvs);

        std::vector<const node *> apply_ops;
        for (auto [v, gs] : var_grads) {
            const auto op = v->apply_gradients(eta, gs);
            apply_ops.push_back(op);
            b.own(op);
        }

        return b.op("minimize",                   //
                    [](basic_runtime<cpu> &) {},  //
                    [](basic_runtime<nvidia_gpu> &) {}, apply_ops);
    }
};
}  // namespace nn::graph::internal
