#pragma once
#include <nn/bits/graph/builder.hpp>

namespace nn::graph::internal
{
class optimizer
{
  public:
    template <typename R, ttl::rank_t r>
    op_node *minimize(builder &b, const var_node<R, r> *l,
                      const float &eta = 0.1) const
    {
        auto [gl, gvs] = b.gradients(l);

        std::map<const base_var_node *, std::vector<const base_var_node *>>
            var_grads;
        for (auto [g, v] : gvs) { var_grads[v].push_back(g); }

        std::vector<const node *> apply_ops;
        for (auto [v, gs] : var_grads) {
            const auto op = v->apply_gradients(eta, gs);
            apply_ops.push_back(op);
            b.own(op);
        }
        return b.op("minimize", [](runtime &) {}, apply_ops);
    }
};
}  // namespace nn::graph::internal
