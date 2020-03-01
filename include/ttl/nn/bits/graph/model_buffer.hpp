#pragma once
#include <map>

#include <ttl/bits/mixed_tensor_buffer.hpp>

namespace ttl::nn::graph::internal
{
template <typename key_t, typename E, typename D>
class basic_model_buffer
{
    using tensor_buffer = ttl::internal::basic_mixed_tensor_buffer<E, D>;
    using symbol_t = typename tensor_buffer::symbol_type;
    std::vector<symbol_t> symbols;

    std::map<key_t, int> index;
    tensor_buffer variables_;

    using raw_tensor_ref = ttl::internal::raw_tensor_ref<E, D>;

  public:
    using symbol_type = symbol_t;

    basic_model_buffer(std::map<key_t, int> index,
                       std::vector<symbol_t> symbols)
        : index(index), variables_(symbols)
    {
    }

    raw_tensor_ref ref(key_t key) { return variables_.ref(index.at(key)); }

    template <typename R, rank_t r>
    tensor_ref<R, r, D> ref(key_t key)
    {
        return ref(key).template typed<R, r>();
    }

    raw_tensor_ref operator[](key_t key) { return ref(key); }
};
}  // namespace ttl::nn::graph::internal
