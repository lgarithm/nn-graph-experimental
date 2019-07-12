#pragma once
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <vector>

#include <nn/bits/graph/back_propagation.hpp>
#include <nn/bits/graph/execution.hpp>
#include <nn/bits/graph/node.hpp>
#include <nn/bits/graph/runtime.hpp>
#include <nn/bits/graph/symbol_manager.hpp>
#include <nn/bits/ops/constant.hpp>
#include <nn/bits/ops/init.hpp>
#include <nn/bits/ops/noop.hpp>
#include <ttl/tensor>

namespace nn::graph::internal
{
template <typename Op, typename... Rs> struct infer {
    using type = typename std::tuple_element<0, std::tuple<Rs...>>::type;
};

class builder
{
  protected:
    std::vector<std::unique_ptr<node>> nodes_;

    std::vector<const base_var_node *> all_vars_;
    std::vector<const base_var_node *> vars_;
    std::vector<const base_var_node *> covars_;
    std::vector<const base_var_node *> tmp_vars_;

    std::vector<const base_func_node *> fns_;

    std::map<const base_var_node *, const base_func_node *> links_;

  public:  // FIXME: make it protected
    void own(const op_node *n)
    {
        nodes_.push_back(std::unique_ptr<node>(const_cast<op_node *>(n)));
    }

  protected:
    void own(const base_func_node *n)
    {
        nodes_.push_back(
            std::unique_ptr<node>(const_cast<base_func_node *>(n)));
        fns_.push_back(n);
    }

    void own(const base_var_node *n)
    {
        nodes_.push_back(std::unique_ptr<node>(const_cast<base_var_node *>(n)));
        all_vars_.push_back(n);
    }

    using task_t = std::function<void(runtime &)>;
    std::vector<task_t> init_ops_;

    template <typename R, ttl::rank_t r>
    var_node<R, r> *tmp_var(const std::string &name, const ttl::shape<r> &shape)
    {
        auto *n = new var_node<R, r>(shape, name);
        own(n);
        tmp_vars_.push_back(n);
        return n;
    }

  public:
    //   alias for ttl::make_shape
    template <typename... D>
    ttl::shape<sizeof...(D)> shape(const D &... d) const
    {
        return ttl::make_shape(d...);
    }

    template <typename R, ttl::rank_t r>
    var_node<R, r> *var(const std::string &name, const ttl::shape<r> &shape)
    {
        auto *n = new var_node<R, r>(shape, name);
        own(n);
        vars_.push_back(n);
        return n;
    }

    template <typename R, ttl::rank_t r, typename Init = nn::ops::noop>
    var_node<R, r> *covar(const std::string &name, const ttl::shape<r> &shape,
                          const Init &init = Init())
    {
        auto *n = new var_node<R, r>(shape, name);
        own(n);
        covars_.push_back(n);
        init_ops_.push_back([=](runtime &rt) { init(n->get_ref(rt)); });
        return n;
    }

    template <typename R, ttl::rank_t r>
    var_node<R, r> *var(const ttl::shape<r> &shape)
    {
        return var<R, r>("", shape);
    }

    template <typename R, ttl::rank_t r, typename Init = nn::ops::noop>
    var_node<R, r> *covar(const ttl::shape<r> &shape, const Init &init = Init())
    {
        return covar<R, r>("", shape, init);
    }

    template <typename R, typename Op, typename... Nodes>
    auto invoke(const std::string &name, const Op &op, const Nodes *... xs)
    {
        const auto shape = op(xs->shape()...);
        constexpr ttl::rank_t r = decltype(shape)::rank;
        using Node = var_node<R, r>;
        Node *y = tmp_var<R>(name, shape);
        const auto fn = demangled_type_info_name(typeid(op));
        auto f = new forward_func_node<Op, Node, Nodes...>(fn, op, y, xs...);
        own(f);
        links_[y] = f;  // TODO: check unique
        return y;
    }

    template <typename Op, typename... Nodes>
    auto invoke(const std::string &name, const Op &op, const Nodes *... xs)
    {
        using R = typename infer<Op, typename Nodes::value_type...>::type;
        return invoke<R>(name, op, xs...);
    }

    template <typename R, typename Op, typename... Nodes>
    auto invoke(const Op &op, const Nodes *... xs)
    {
        return invoke<R>("", op, xs...);
    }

    template <typename Op, typename... Nodes>
    auto invoke(const Op &op, const Nodes *... xs)
    {
        using R = typename infer<Op, typename Nodes::value_type...>::type;
        return invoke<R>(op, xs...);
    }

    op_node *op(const std::string &name,
                const std::function<void(runtime &)> &f,
                const std::vector<const node *> &deps = {})
    {
        auto n = new op_node(name, f, deps);
        own(n);
        return n;
    }

    using grad_var_t = std::pair<const base_var_node *, const base_var_node *>;

    template <typename R, ttl::rank_t r> auto gradients(const var_node<R, r> *y)
    {
        const std::set<const base_var_node *> xs(covars_.begin(),
                                                 covars_.end());

        std::queue<grad_var_t> q;
        using grad_init_t =
            typename grad_init<std::is_floating_point<R>::value>::type;
        auto gy = covar<R>("g_" + y->name(), y->shape(), grad_init_t());
        q.push(std::make_pair(gy, y));

        std::vector<grad_var_t> gvs;
        while (!q.empty()) {
            const auto [gy, y] = q.front();
            q.pop();
            if (links_.count(y) > 0) {
                const auto f = links_.at(y);
                for (const auto [gxi, gi, xi] : f->all_gradients(gy)) {
                    q.push(std::make_pair(gxi, xi));
                    own(gi);
                    own(gxi);
                    tmp_vars_.push_back(gxi);
                    links_[gxi] = gi;  // TODO: check unique
                }
            } else {
                if (xs.count(y) > 0) { gvs.push_back(std::make_pair(gy, y)); }
            }
        }
        return std::make_pair(gy, gvs);
    }

    void debug() const
    {
        std::cerr << all_vars_.size() << " vars" << std::endl;
        for (const auto v : all_vars_) {
            std::cerr << "  " << v->name() << " :: " << v->str();
            if (links_.count(v) == 0) { std::cerr << " *"; }
            std::cerr << std::endl;
        }
        std::cerr << fns_.size() << " fns:" << std::endl;
        for (const auto f : fns_) {
            auto y = f->output();
            auto xs = f->inputs();
            std::stringstream ss;
            std::transform(xs.begin(), xs.end(),
                           std::experimental::make_ostream_joiner(ss, ", "),
                           [](auto v) { return v->name(); });
            std::cerr << "  " << y->name() << " = " << f->name() << "("
                      << ss.str() << ")" << std::endl;
        }
        std::cerr << links_.size() << " links:" << std::endl;
        for (const auto [y, f] : links_) {
            std::cerr << "  " << y->name() << " <- " << f->name() << std::endl;
        }
    }

    void build(runtime &rt) const
    {
        for (const auto v : covars_) { v->create(rt); }
        for (const auto v : tmp_vars_) { v->create(rt); }
        for (const auto v : vars_) { v->define(rt); }
    }

    void init(runtime &rt) const
    {
        for (const auto init : init_ops_) { init(rt); }
    }

    // void run(runtime &rt, const node *y) const
    // {
    //     std::cerr << "[W] TODO:" << std::endl;
    // }

    void _run(runtime &rt, execution &e, const base_func_node *f) const
    {
        if (e.is_done(f)) { return; }
        for (const auto x : f->inputs()) {
            if (links_.count(x) > 0) { _run(rt, e, links_.at(x)); }
        }
        f->run(rt);
        e.done(f);
    }

    void run(runtime &rt, const std::vector<const base_var_node *> &ys) const
    {
        execution e;
        for (const auto y : ys) {
            if (links_.count(y) > 0) { _run(rt, e, links_.at(y)); }
        }
    }

    void run(runtime &rt, const base_var_node *y) const
    {
        run(rt, std::vector<const base_var_node *>({y}));
    }

    void _run(runtime &rt, execution &e, const op_node *y) const
    {
        if (e.is_done(y)) { return; }
        for (const auto x : y->dependencies()) {
            const op_node *p_op = dynamic_cast<const op_node *>(x);
            const base_var_node *p_var = dynamic_cast<const base_var_node *>(x);
            if (p_op) {
                _run(rt, e, p_op);
            } else if (p_var) {
                if (links_.count(p_var) > 0) { _run(rt, e, links_.at(p_var)); }
            } else {
                std::cerr << "_run undefined for node" << std::endl;
            }
        }
        y->run(rt);
    }

    void run(runtime &rt, const op_node *y) const
    {
        execution e;
        _run(rt, e, y);
    }
};

}  // namespace nn::graph::internal
