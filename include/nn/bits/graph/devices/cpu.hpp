#pragma once
#include <ttl/tensor>

namespace nn::graph::internal
{
struct cpu {
    template <typename R, ttl::rank_t r>  //
    using tensor_type = ttl::tensor<R, r>;

    template <typename R, ttl::rank_t r>
    using reference_type = ttl::tensor_ref<R, r>;

    template <typename R, ttl::rank_t r>
    using view_type = ttl::tensor_view<R, r>;
};
};  // namespace nn::graph::internal
