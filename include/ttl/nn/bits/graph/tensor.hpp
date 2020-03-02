#pragma once
#include <ttl/experimental/raw_tensor>

namespace ttl::nn::graph::internal
{
using flat_shape = ttl::internal::basic_flat_shape<>;

// FIXME: use std encoding
using idx_encoder =
    ttl::internal::basic_type_encoder<ttl::internal::idx_format::encoding>;

template <typename D>
using raw_tensor = ttl::internal::basic_raw_tensor<
    idx_encoder, ttl::internal::basic_flat_shape<>, D, ttl::internal::owner>;

template <typename D>
using raw_tensor_ref =
    ttl::internal::basic_raw_tensor<idx_encoder,
                                    ttl::internal::basic_flat_shape<>, D,
                                    ttl::internal::readwrite>;

template <typename D>
using raw_tensor_view = ttl::internal::basic_raw_tensor<
    idx_encoder, ttl::internal::basic_flat_shape<>, D, ttl::internal::readonly>;
}  // namespace ttl::nn::graph::internal
