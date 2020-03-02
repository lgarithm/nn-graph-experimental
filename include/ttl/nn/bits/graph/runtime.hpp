#pragma once
#include <map>

#include <ttl/experimental/raw_tensor>
#include <ttl/experimental/show_size>
#include <ttl/nn/bits/graph/apply.hpp>
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/nn/bits/graph/model_buffer.hpp>
#include <ttl/nn/bits/graph/variable_manager.hpp>
#include <ttl/nn/bits/tuple.hpp>
#include <ttl/shape>

namespace ttl::nn::graph::internal
{
class base_var_node;

template <typename R, rank_t r>
class var_node;

template <typename RT>
struct _get_ref {
    const RT *rt;

    _get_ref(const RT *rt) : rt(rt) {}

    template <typename Node>
    auto operator()(const Node *key) const
    {
        using R = typename Node::value_type;
        constexpr auto r = Node::rank;
        return rt->template ref<R, r>(key);
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
        return rt->template view<R, r>(key);
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
    using idx_encoder =
        ttl::internal::basic_type_encoder<ttl::internal::idx_format::encoding>;

  public:
    using model_buffer_t = basic_model_buffer<key_t, idx_encoder, D>;

  protected:
    std::unique_ptr<model_buffer_t> mb_;
    variable_manager<D> vm_;

    std::map<key_t, variable<D> *> vars_;
    std::map<key_t, reference<D> *> binds_;

    template <typename R, rank_t r>
    tensor_ref<R, r, D> get(key_t key) const
    {
        if (binds_.count(key) > 0) {
            return binds_.at(key)->ref().template typed<R, r>();
        } else if (vars_.count(key) > 0) {
            return vars_.at(key)->ref().template typed<R, r>();
        } else {
            return mb_->template ref<R, r>(key);
        }
    }

  public:
    using device_type = D;
    static constexpr D device = default_device<D>::value;

    void set_model(model_buffer_t *mb) { mb_.reset(mb); }

    auto create(const tensor_symbol &sym, key_t key)
    {
        if (vars_.count(key) > 0) {
            throw std::logic_error("duplicated creation");
        }
        auto t = vm_.create_tensor(sym);
        vars_[key] = t;
        return t;
    }

    auto define(const tensor_symbol &sym, key_t key)
    {
        if (binds_.count(key) > 0) {
            throw std::logic_error("duplicated definition");
        }
        auto t = vm_.create_tensor_reference(sym);
        binds_[key] = t;
        return t;
    }

    template <typename R, rank_t r>
    void bind(key_t key, const tensor_ref<R, r, D> &t)
    {
        tensor_view<R, r, D> vv(t);
        raw_tensor_view<D> v(vv);
        binds_.at(key)->bind(v);
    }

    template <typename R, rank_t r>
    void bind(key_t key, const tensor_view<R, r, D> &t)
    {
        // FIXME: !
        tensor_ref<R, r, D> x(const_cast<R *>(t.data()), t.shape());
        bind(key, x);
    }

    void unbind(key_t key) { binds_.at(key)->unbind(); }

    template <typename R, rank_t r>
    tensor_ref<R, r, D> ref(key_t key) const
    {
        return get<R, r>(key);
    }

    template <typename R, rank_t r>
    tensor_view<R, r, D> view(key_t key) const
    {
        return get<R, r>(key);
    }

    raw_tensor_ref<D> ref(key_t key) const
    {
        // FIXME: handle ttl::cuda_memory
        if (binds_.count(key) > 0) {
            return binds_.at(key)->ref();
        } else if (vars_.count(key) > 0) {
            return vars_.at(key)->ref();
        } else {
            return mb_->ref(key);
        }
    }

    template <typename R, rank_t r>
    tensor_ref<R, r, D> ref(const var_node<R, r> *key) const
    {
        return get<R, r>(key);
    }

    template <typename R, rank_t r>
    tensor_view<R, r, D> view(const var_node<R, r> *key) const
    {
        return get<R, r>(key);
    }

    raw_tensor_view<D> view(key_t key) const
    {
        return raw_tensor_view<D>(ref(key));
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
