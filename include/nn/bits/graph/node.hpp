#pragma once
#include <iostream>
#include <optional>
#include <vector>

#include <nn/bits/graph/common.hpp>
#include <nn/bits/graph/execution.hpp>
#include <nn/bits/graph/runtime.hpp>
#include <nn/bits/ops/axpy.hpp>
#include <nn/bits/tuple.hpp>
#include <nn/gradients>
#include <ttl/tensor>

namespace nn::graph::internal
{

class node
{
  public:
    virtual ~node() {}

    virtual const std::string &name() const = 0;

    virtual void run(runtime &rt) const = 0;
};

class op_node : public node
{
    const std::string name_;

    using operation_t = std::function<void(runtime &)>;
    const operation_t f_;

    const std::vector<const node *> dependencies_;

  public:
    op_node(const std::string &name, const operation_t &f,
            const std::vector<const node *> &deps = {})
        : name_(name), f_(f), dependencies_(deps)
    {
    }

    const std::string &name() const override { return name_; }

    void run(runtime &rt) const override { f_(rt); }

    std::vector<const node *> dependencies() const { return dependencies_; }
};

template <typename R, ttl::rank_t r> class var_node;

class base_var_node : public node
{
  public:
    virtual ~base_var_node() {}

    virtual base_var_node *dup(const std::string &name) const = 0;

    virtual std::string str() const = 0;

    virtual void create(runtime &rt) const = 0;

    virtual void define(runtime &rt) const = 0;

    template <typename R, ttl::rank_t r> var_node<R, r> *as() const
    {
        using V = var_node<R, r>;
        return const_cast<V *>(down_cast<V>(this));
    }

    virtual op_node *
    apply_gradients(const float &eta,
                    const std::vector<const base_var_node *> &gs) const = 0;
};

template <typename R, ttl::rank_t r> class var_node : public base_var_node
{
    const ttl::shape<r> shape_;
    const std::string name_;

  public:
    using value_type = R;
    static constexpr ttl::rank_t rank = r;

    var_node(const var_node &) = delete;

    var_node(const ttl::shape<r> &shape, const std::string &name)
        : shape_(shape), name_(name)
    {
    }

    const std::string &name() const override { return name_; }

    std::string str() const override
    {
        return scalar_type_name().template operator()<R>() +
               ttl::to_string(shape_);
    }

    void run(runtime &rt) const override {}  // noop

    var_node *dup(const std::string &name) const override
    {
        return new var_node(shape_, name);
    }

    void create(runtime &rt) const override { rt.create<R>(shape_, this); }

    void define(runtime &rt) const override { rt.define<R>(shape_, this); }

    ttl::shape<r> shape() const { return shape_; }

    ttl::tensor_ref<R, r> get_ref(runtime &rt) const
    {
        return rt.get<R, r>(this);
    }

    ttl::tensor_view<R, r> get_view(runtime &rt) const
    {
        return rt.get<R, r>(this);
    }

    op_node *
    apply_gradients(const float &eta,
                    const std::vector<const base_var_node *> &gs) const override
    {
        std::vector<const node *> deps(gs.size());
        std::copy(gs.begin(), gs.end(), deps.begin());
        return new op_node("apply_grad",
                           [&, eta = eta, gs = gs, name = name_](runtime &rt) {
                               ttl::tensor<R, 0> a;
                               a.data()[0] = -eta;
                               for (const auto g : gs) {
                                   nn::ops::axpy()(get_ref(rt), view(a),
                                                   g->as<R, r>()->get_view(rt),
                                                   get_view(rt));
                               }
                           },
                           deps);
    }
};

class base_func_node : public node
{
  protected:
    using inputs_t = std::vector<const base_var_node *>;

  public:
    virtual ~base_func_node() {}

    virtual inputs_t inputs() const = 0;

    virtual const base_var_node *output() const = 0;

    virtual std::vector<std::tuple<
        const base_var_node *, const base_func_node *, const base_var_node *>>
    all_gradients(const base_var_node *gy) const = 0;
};

class get_view
{
    runtime &rt_;

  public:
    get_view(runtime &rt) : rt_(rt) {}

    template <typename Node> auto operator()(const Node *x) const
    {
        return x->get_view(rt_);
    }
};

template <typename F, typename Node, typename... Nodes>
class func_node : public base_func_node
{
  protected:
    const std::string name_;
    const F f_;
    const Node *y_;
    const std::tuple<const Nodes *...> xs_;

  public:
    using value_type = typename Node::value_type;

    static constexpr auto arity = sizeof...(Nodes);

    func_node(const std::string &name, const F &f, const Node *y,
              const std::tuple<const Nodes *...> &xs)
        : name_(name), f_(f), y_(y), xs_(xs)
    {
    }

    const std::string &name() const override { return name_; }

    inputs_t inputs() const override
    {
        inputs_t is;
        for (const auto x : tup2arr<const base_var_node *>(xs_)) {
            is.push_back(x);
        }
        return is;
    }

    const base_var_node *output() const override { return y_; }

    void run(runtime &rt) const override
    {
        const auto inputs = tuple_map(get_view(rt), xs_);
        const auto outputs = std::make_tuple(y_->get_ref(rt));
        std::apply(f_, std::tuple_cat(outputs, inputs));
    }
};

}  // namespace nn::graph::internal
