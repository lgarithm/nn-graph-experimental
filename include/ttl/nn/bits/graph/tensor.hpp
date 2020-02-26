#pragma once
#include <ttl/experimental/raw_tensor>

namespace ttl::nn::graph::internal
{
using flat_shape = ttl::internal::basic_flat_shape<>;

// FIXME: use std encoding
using idx_encoder =
    std::experimental::basic_type_encoder<ttl::internal::idx_format::encoding>;

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

template <typename E, typename S = ttl::internal::basic_flat_shape<>>
class basic_raw_tensor_symbol
{
    using value_type_t = typename E::value_type;

    const value_type_t value_type_;
    const S shape_;

  public:
    using encoder_type = E;
    using shape_type = S;

    template <typename R>
    static constexpr value_type_t type()
    {
        return E::template value<R>();
    }

    basic_raw_tensor_symbol(const value_type_t &value_type, const S &shape)
        : value_type_(value_type), shape_(shape)
    {
    }

    value_type_t value_type() const { return value_type_; }

    const S &shape() const { return shape_; }

    size_t data_size() const { return E::size(value_type_) * shape_.size(); }
};

using tensor_symbol = basic_raw_tensor_symbol<idx_encoder>;
}  // namespace ttl::nn::graph::internal
