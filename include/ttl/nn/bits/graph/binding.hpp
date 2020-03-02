#pragma once
#include <ttl/tensor>

namespace ttl::nn::graph::internal
{
class base_var_node;

template <typename RT>
class binding
{
    using D = typename RT::device_type;
    using key_t = const base_var_node *;

    RT &rt;
    key_t key;

  public:
    template <typename R, rank_t r>
    binding(RT &rt, key_t key, const tensor_view<R, r, D> &x) : rt(rt), key(key)
    {
        rt.bind(key, x);
    }

    ~binding() { rt.unbind(key); }
};
}  // namespace ttl::nn::graph::internal
