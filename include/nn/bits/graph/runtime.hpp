#pragma once
#include <map>

#include <ttl/experimental/raw_tensor>
#include <ttl/experimental/show_size>
#include <ttl/shape>

#include <nn/bits/graph/apply.hpp>
#include <nn/bits/graph/variable_manager.hpp>
#include <nn/bits/tuple.hpp>

namespace ttl::nn::graph::internal
{
class base_var_node;

template <typename RT>
struct _get_ref {
    const RT *rt;

    _get_ref(const RT *rt) : rt(rt) {}

    template <typename Node>
    auto operator()(const Node *key) const
    {
        using R = typename Node::value_type;
        constexpr auto r = Node::rank;
        return rt->template get_ref<R, r>(key);
    }
};

template <typename RT>
struct _get_view {
    const RT *rt;

    _get_view(const RT *rt) : rt(rt) {}

    template <typename Node>
    auto operator()(const Node *key) const
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

template <typename D>
class basic_runtime : public runtime
{
  protected:
    variable_manager<D> vm_;

    std::map<key_t, variable<D> *> vars_;
    std::map<key_t, reference<D> *> binds_;

    template <typename R, rank_t r>
    ttl::tensor_ref<R, r, D> get(key_t key) const
    {
        if (binds_.count(key) > 0) {
            return binds_.at(key)->template as<R, r>().get();
        } else {
            return vars_.at(key)->template as<R, r>().get();
        }
    }

  public:
    template <typename R, rank_t r>
    auto create(const ttl::shape<r> &shape, key_t key)
    {
        if (vars_.count(key) > 0) {
            throw std::logic_error("duplicated creation");
        }
        auto t = vm_.template create_tensor<R>(shape);
        vars_[key] = t;
        return t;
    }

    template <typename R, rank_t r>
    auto define(const ttl::shape<r> &shape, key_t key)
    {
        if (binds_.count(key) > 0) {
            throw std::logic_error("duplicated definition");
        }
        auto t = vm_.template create_tensor_reference<R, r>(shape);
        binds_[key] = t;
        return t;
    }

    template <typename R, rank_t r>
    void bind(key_t key, const ttl::tensor_ref<R, r, D> &t)
    {
        binds_.at(key)->template as<R, r>().bind(t);
    }

    void unbind(key_t key) { binds_.at(key)->unbind(); }

    template <typename R, rank_t r>
    ttl::tensor_ref<R, r, D> get_ref(key_t key) const
    {
        return get<R, r>(key);
    }

    template <typename R, rank_t r>
    ttl::tensor_view<R, r, D> get_view(key_t key) const
    {
        return get<R, r>(key);
    }

    raw_tensor_ref<D> get_raw_ref(key_t key) const
    {
        // FIXME: handle ttl::cuda_memory
        if (binds_.count(key) > 0) {
            return binds_.at(key)->raw_ref();
        } else {
            return vars_.at(key)->raw_ref();
        }
    }

    raw_tensor_view<D> get_raw_view(key_t key) const
    {
        return raw_tensor_view<D>(get_raw_ref(key));
    }

    template <typename F, typename Outputs, typename Inputs>
    void run(const F &f, const Outputs &outputs, const Inputs &inputs) const
    {
        const auto ys = tuple_map(_get_ref(this), outputs);
        const auto xs = tuple_map(_get_view(this), inputs);
        apply_if<D>(f, std::tuple_cat(ys, xs));
    }

    // debug

    void debug()
    {
        size_t tot = 0;
        for (auto [_, v] : vars_) {
            static_assert(sizeof(_) > 0, "");
            std::cerr << static_cast<std::string>(*v) << std::endl;
            tot += v->data_size();
        }
        std::cerr << "total size: " << show_size(tot) << std::endl;
    }
};

using cpu_runtime = basic_runtime<cpu>;
using gpu_runtime = basic_runtime<nvidia_gpu>;
}  // namespace ttl::nn::graph::internal
