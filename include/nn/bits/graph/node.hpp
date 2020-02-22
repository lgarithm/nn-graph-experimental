#pragma once
#include <iostream>
#include <optional>
#include <vector>

#include <nn/bits/graph/common.hpp>
#include <nn/bits/graph/cuda_ops.hpp>
#include <nn/bits/graph/device.hpp>
#include <nn/bits/graph/execution.hpp>
#include <nn/bits/graph/operator.hpp>
#include <nn/bits/graph/runtime.hpp>
#include <nn/gradients>
#include <ttl/cuda_tensor>
#include <ttl/tensor>

namespace ttl::nn::graph::internal
{

class node
{
  public:
    virtual ~node() {}

    virtual const std::string &name() const = 0;

    virtual void run(cpu_runtime &rt) const = 0;

    virtual void run(gpu_runtime &rt) const = 0;
};

class op_node : public node
{
    const std::string name_;

    using cpu_operation_t = std::function<void(cpu_runtime &)>;

    using gpu_operation_t = std::function<void(gpu_runtime &)>;

    const cpu_operation_t f_cpu_;
    const gpu_operation_t f_gpu_;

    const std::vector<const node *> dependencies_;

  public:
    op_node(const std::string &name, const cpu_operation_t &f,
            const std::vector<const node *> &deps = {})
        : name_(name), f_cpu_(f), dependencies_(deps)
    {
    }

    op_node(const std::string &name, const gpu_operation_t &f,
            const std::vector<const node *> &deps = {})
        : name_(name), f_gpu_(f), dependencies_(deps)
    {
    }

    op_node(const std::string &name, const cpu_operation_t &f,
            const gpu_operation_t &g,
            const std::vector<const node *> &deps = {})
        : name_(name), f_cpu_(f), f_gpu_(g), dependencies_(deps)
    {
    }

    const std::string &name() const override { return name_; }

    void run(cpu_runtime &rt) const override { f_cpu_(rt); }

    void run(gpu_runtime &rt) const override { f_gpu_(rt); }

    std::vector<const node *> dependencies() const { return dependencies_; }
};

template <typename R, ttl::rank_t r>
class var_node;

class base_var_node : public node
{
  protected:
    using var_node_list_t = std::vector<const base_var_node *>;

  public:
    virtual ~base_var_node() {}

    virtual base_var_node *dup(const std::string &name) const = 0;

    virtual std::string str() const = 0;

    virtual void create(cpu_runtime &rt) const = 0;

    virtual void define(cpu_runtime &rt) const = 0;

    virtual void create(gpu_runtime &rt) const = 0;

    virtual void define(gpu_runtime &rt) const = 0;

    template <typename R, ttl::rank_t r>
    var_node<R, r> *as() const
    {
        using V = var_node<R, r>;
        return const_cast<V *>(down_cast<V>(this));
    }
};

template <typename R, ttl::rank_t r>
class var_node : public base_var_node
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

    void run(cpu_runtime &rt) const override {}  // noop

    void run(gpu_runtime &rt) const override {}  // noop

    var_node *dup(const std::string &name) const override
    {
        return new var_node(shape_, name);
    }

    void create(cpu_runtime &rt) const override { rt.create<R>(shape_, this); }

    void define(cpu_runtime &rt) const override { rt.define<R>(shape_, this); }

    void create(gpu_runtime &rt) const override { rt.create<R>(shape_, this); }

    void define(gpu_runtime &rt) const override { rt.define<R>(shape_, this); }

    ttl::shape<r> shape() const { return shape_; }

    ttl::tensor_ref<R, r> get_ref(cpu_runtime &rt) const
    {
        return rt.get_ref<R, r>(this);
    }

    ttl::tensor_view<R, r> get_view(cpu_runtime &rt) const
    {
        return rt.get_view<R, r>(this);
    }

    ttl::cuda_tensor_ref<R, r> get_ref(gpu_runtime &rt) const
    {
        return rt.get_ref<R, r>(this);
    }

    ttl::cuda_tensor_view<R, r> get_view(gpu_runtime &rt) const
    {
        return rt.get_view<R, r>(this);
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

template <typename F, typename Node, typename... Nodes>
class func_node : public base_func_node
{
    using gpu_op_t = typename for_device<F, nvidia_gpu>::type;

  protected:
    const std::string name_;
    const F f_;
    const gpu_op_t g_;

    const Node *y_;
    const std::tuple<const Nodes *...> xs_;

  public:
    using value_type = typename Node::value_type;

    static constexpr auto arity = sizeof...(Nodes);

    func_node(const std::string &name, const F &f, const Node *y,
              const std::tuple<const Nodes *...> &xs)
        : name_(name), f_(f), g_(create_op<gpu_op_t>(f)), y_(y), xs_(xs)
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

    void run(cpu_runtime &rt) const override
    {
        rt.run(f_, std::make_tuple(y_), xs_);
    }

    void run(gpu_runtime &rt) const override
    {
        rt.run(g_, std::make_tuple(y_), xs_);
    }
};
}  // namespace ttl::nn::graph::internal
