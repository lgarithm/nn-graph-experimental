#pragma once
#include <map>

#include <ttl/debug>
#include <ttl/tensor>

#include <nn/bits/graph/variable_manager.hpp>

namespace nn::graph::internal
{
class base_var_node;

// runtime keeps the persistent state
class runtime
{
    variable_manager vm_;

    using key_t = const base_var_node *;

    std::map<key_t, variable *> vars_;
    std::map<key_t, reference *> binds_;

  public:
    template <typename R, ttl::rank_t r>
    auto create(const ttl::shape<r> &shape, key_t key)
    {
        if (vars_.count(key) > 0) {
            throw std::logic_error("duplicated creation");
        }
        // std::cerr << "creating " << key << " :: " << ttl::to_string(shape)
        //           << std::endl;
        auto t = vm_.create_tensor<R>(shape);
        vars_[key] = t;
        return t;
    }

    template <typename R, ttl::rank_t r>
    auto define(const ttl::shape<r> &shape, key_t key)
    {
        // std::cerr << "define " << key << " :: " << ttl::to_string(shape)
        //           << " | " << name << std::endl;
        if (binds_.count(key) > 0) {
            throw std::logic_error("duplicated definition");
        }
        auto t = vm_.create_tensor_reference<R, r>(shape);
        binds_[key] = t;
        return t;
    }

    template <typename R, ttl::rank_t r>
    void bind(key_t key, const ttl::tensor_ref<R, r> &t)
    {
        // std::cerr << "binding " << key << " <- " << ttl::to_string(t.shape())
        //           << std::endl;
        binds_.at(key)->as<R, r>().bind(t);
    }

    void unbind(key_t key) { binds_.at(key)->unbind(); }

    template <typename R, ttl::rank_t r> ttl::tensor_ref<R, r> get(key_t key)
    {
        if (binds_.count(key) > 0) {
            return binds_.at(key)->as<R, r>().get();
        } else {
            return vars_.at(key)->as<R, r>().get();
        }
    }
};
}  // namespace nn::graph::internal
