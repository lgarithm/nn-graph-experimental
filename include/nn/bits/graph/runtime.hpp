#pragma once
#include <map>

#include <ttl/shape>
#include <ttl/show_size>

#include <nn/bits/graph/apply.hpp>
#include <nn/bits/graph/devices/cpu.hpp>
#include <nn/bits/graph/devices/nvidia_gpu.hpp>
#include <nn/bits/graph/variable_manager.hpp>
#include <nn/bits/tuple.hpp>

namespace nn::graph::internal
{
class base_var_node;

template <typename RT> struct _get_ref {
    const RT *rt;

    _get_ref(const RT *rt) : rt(rt) {}

    template <typename Node> auto operator()(const Node *key) const
    {
        using R = typename Node::value_type;
        constexpr auto r = Node::rank;
        return rt->template get_ref<R, r>(key);
    }
};

template <typename RT> struct _get_view {
    const RT *rt;

    _get_view(const RT *rt) : rt(rt) {}

    template <typename Node> auto operator()(const Node *key) const
    {
        using R = typename Node::value_type;
        constexpr auto r = Node::rank;
        return rt->template get_view<R, r>(key);
    }
};

// runtime keeps the persistent state
class runtime
{
  protected:
    using key_t = const base_var_node *;

  public:
};

template <typename device> class basic_runtime : public runtime
{
  public:
    template <typename R, ttl::rank_t r>
    using ref_t = typename device::template reference_type<R, r>;

    template <typename R, ttl::rank_t r>
    using view_t = typename device::template view_type<R, r>;

  protected:
    variable_manager<device> vm_;

    std::map<key_t, variable *> vars_;
    std::map<key_t, reference *> binds_;

    template <typename R, ttl::rank_t r> ref_t<R, r> get(key_t key) const
    {
        if (binds_.count(key) > 0) {
            return binds_.at(key)->template as<R, r, device>().get();
        } else {
            return vars_.at(key)->template as<R, r, device>().get();
        }
    }

  public:
    template <typename R, ttl::rank_t r>
    auto create(const ttl::shape<r> &shape, key_t key)
    {
        if (vars_.count(key) > 0) {
            throw std::logic_error("duplicated creation");
        }
        auto t = vm_.template create_tensor<R>(shape);
        vars_[key] = t;
        return t;
    }

    template <typename R, ttl::rank_t r>
    auto define(const ttl::shape<r> &shape, key_t key)
    {
        if (binds_.count(key) > 0) {
            throw std::logic_error("duplicated definition");
        }
        auto t = vm_.template create_tensor_reference<R, r>(shape);
        binds_[key] = t;
        return t;
    }

    template <typename R, ttl::rank_t r>
    void bind(key_t key, const ref_t<R, r> &t)
    {
        binds_.at(key)->template as<R, r, device>().bind(t);
    }

    void unbind(key_t key) { binds_.at(key)->unbind(); }

    template <typename R, ttl::rank_t r> ref_t<R, r> get_ref(key_t key) const
    {
        return get<R, r>(key);
    }

    template <typename R, ttl::rank_t r> view_t<R, r> get_view(key_t key) const
    {
        return get<R, r>(key);
    }

    template <typename F, typename Outputs, typename Inputs>
    void run(const F &f, const Outputs &outputs, const Inputs &inputs) const
    {
        const auto ys = tuple_map(_get_ref(this), outputs);
        const auto xs = tuple_map(_get_view(this), inputs);
        apply_if<device>(f, std::tuple_cat(ys, xs));
    }

    // debug

    void debug()
    {
        size_t tot = 0;
        for (auto [_, v] : vars_) {
            std::cerr << static_cast<std::string>(*v) << std::endl;
            tot += v->data_size();
        }
        std::cerr << "total size: " << show_size(tot) << std::endl;
    }
};

using cpu_runtime = basic_runtime<cpu>;
using gpu_runtime = basic_runtime<nvidia_gpu>;
}  // namespace nn::graph::internal
