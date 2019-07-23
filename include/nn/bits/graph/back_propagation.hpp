#pragma once
#include <nn/bits/graph/node.hpp>
#include <nn/bits/ops/constant.hpp>
#include <nn/bits/ops/cuda_ops.hpp>
#include <nn/bits/ops/init.hpp>

namespace nn::graph::internal
{
template <bool> struct grad_init;

template <> struct grad_init<true> {
    using type = nn::ops::uniform_distribution;
};

template <> struct grad_init<false> {
    using type = nn::ops::ones;
};

template <arity_t i, typename Gi, typename Node, typename... Nodes>
class gard_func_node
    : public func_node<
          Gi, typename std::tuple_element<i, std::tuple<Nodes...>>::type, Node,
          Node, Nodes...>
{
    using Xi = typename std::tuple_element<i, std::tuple<Nodes...>>::type;
    using FN = func_node<Gi, Xi, Node, Node, Nodes...>;

  public:
    gard_func_node(const std::string &name, const Gi &op, const Xi *gxi,
                   const Node *gy, const Node *y,
                   const std::tuple<const Nodes *...> &xs)
        : FN(name, op, gxi, std::tuple_cat(std::make_tuple(gy, y), xs))
    {
    }

    std::vector<std::tuple<const base_var_node *, const base_func_node *,
                           const base_var_node *>>
    all_gradients(const base_var_node *gy) const override
    {
        throw std::logic_error(
            "gard_func_node::all_gradients shouldn't be called");
    }
};

template <bool, class G, class F> struct create_grad_op;
template <class G, class F> struct create_grad_op<false, G, F> {
    G operator()(const F &f) const
    {
        G g;
        return g;
    }
};

template <class G, class F> struct create_grad_op<true, G, F> {
    G operator()(const F &f) const { return G(f); }
};

template <bool, int i, class Gi, class F, class Xi, class Node, class... Nodes>
struct create_grad_func_node;

template <int i, class Gi, class F, class Xi, class Node, class... Nodes>
struct create_grad_func_node<false, i, Gi, F, Xi, Node, Nodes...> {
    base_func_node *operator()(const Xi *gxi, const Node *gy, const F &f,
                               const Node *y,
                               const std::tuple<const Nodes *...> &xs) const
    {
        if (gxi) {
            std::cerr << "[W] F has no gradient " << static_cast<int>(i)
                      << " but gx[i] is allocated" << std::endl;
        }
        return nullptr;
    }
};

template <int i, class Gi, class F, class Xi, class Node, class... Nodes>
struct create_grad_func_node<true, i, Gi, F, Xi, Node, Nodes...> {
    base_func_node *operator()(const Xi *gxi, const Node *gy, const F &f,
                               const Node *y,
                               const std::tuple<const Nodes *...> &xs) const
    {
        static_assert(
            std::is_same<Xi, typename std::tuple_element<
                                 i, std::tuple<Nodes...>>::type>::value);

        if (gxi == nullptr) { throw std::logic_error("gx[i] is nullptr"); }
        const auto gn = demangled_type_info_name(typeid(Gi));
        const Gi gi =
            create_grad_op<std::is_constructible<Gi, F>::value, Gi, F>()(f);
        return new gard_func_node<i, Gi, Node, Nodes...>(gn, gi, gxi, gy, y,
                                                         xs);
    }
};

template <typename F, typename Node, typename... Nodes>
class forward_func_node : public func_node<F, Node, Nodes...>
{
    using FN = func_node<F, Node, Nodes...>;

    template <arity_t i>
    base_func_node *gradient(const base_var_node *gxi, const Node *gy) const
    {
        using Gi = typename nn::gradient<F, i>::type;
        using Xi = typename std::tuple_element<i, std::tuple<Nodes...>>::type;
        using create_node =
            create_grad_func_node<std::negation<std::is_same<Gi, void>>::value,
                                  i, Gi, F, Xi, Node, Nodes...>;
        // gxi can be nullptr
        return create_node()(dynamic_cast<const Xi *>(gxi), gy, this->f_,
                             this->y_, this->xs_);
    }

    template <size_t... I>
    auto _all_gradients(
        std::index_sequence<I...>, const Node *gy,
        const std::array<const base_var_node *, FN::arity> &gxs) const
    {
        return std::vector<const base_func_node *>(
            {gradient<I>(std::get<I>(gxs), gy)...});
    }

  public:
    forward_func_node(const std::string &name, const F &op, const Node *y,
                      const Nodes *... xs)
        : FN(name, op, y, std::make_tuple(xs...))
    {
    }

    template <size_t... I>
    constexpr auto is_differentiable(std::index_sequence<I...>) const
    {
        return std::array<bool, FN::arity>({std::negation<typename std::is_same<
            typename nn::gradient<F, I>::type, void>>::value...});
    }

    std::vector<std::tuple<const base_var_node *, const base_func_node *,
                           const base_var_node *>>
    all_gradients(const base_var_node *gy) const override
    {
        const auto mask =
            is_differentiable(std::make_index_sequence<FN::arity>());
        std::vector<const base_var_node *> xs = this->inputs();
        std::vector<const base_var_node *> gxs(FN::arity);
        for (auto i : range(FN::arity)) {
            if (mask[i]) {
                gxs[i] = xs[i]->dup("g_" + xs[i]->name());
            } else {
                gxs[i] = nullptr;
            }
        }

        auto gs = _all_gradients(std::make_index_sequence<FN::arity>(),
                                 down_cast<Node>(gy), vec2arr<FN::arity>(gxs));

        std::vector<std::tuple<const base_var_node *, const base_func_node *,
                               const base_var_node *>>
            gfv;
        for (auto i : range(FN::arity)) {
            if (mask[i]) {
                gfv.push_back(std::make_tuple(gxs[i], gs[i], xs[i]));
            }
        }
        return gfv;
    }
};

}  // namespace nn::graph::internal
