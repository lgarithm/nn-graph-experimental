#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ttl/bits/std_tensor_symbol.hpp>
#include <ttl/debug>
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/nn/bits/graph/tensor.hpp>
#include <ttl/shape>
#include <ttl/tensor>

namespace ttl::nn::graph::internal
{
using tensor_symbol = ttl::internal::basic_raw_tensor_symbol<idx_encoder>;

template <typename D>
class variable
{
    using E = typename raw_tensor<D>::encoder_type;
    using value_type_t = typename E::value_type;
    const raw_tensor<D> value_;

  public:
    variable(const value_type_t &type, const flat_shape &shape)
        : value_(type, shape)
    {
    }

    variable(const tensor_symbol &sym) : variable(sym.value_type(), sym.shape())
    {
    }

    raw_tensor_ref<D> ref() const { return raw_tensor_ref<D>(value_); }

    raw_tensor_view<D> view() const { return raw_tensor_view<D>(value_); }

    size_t data_size() const { return value_.data_size(); }

    operator std::string() const { return "T" + to_string(value_.shape()); }
};

template <typename D>
class reference
{
    using E = typename raw_tensor<D>::encoder_type;
    using value_type_t = typename E::value_type;

    const value_type_t value_type_;
    const flat_shape shape_;
    std::optional<raw_tensor_ref<D>> value_;

  public:
    reference(const value_type_t &type, const flat_shape &shape)
        : value_type_(type), shape_(shape)
    {
    }

    reference(const tensor_symbol &sym)
        : reference(sym.value_type(), sym.shape())
    {
    }

    void bind(const raw_tensor_view<D> &t)
    {
        if (t.shape().size() != shape_.size()) {
            throw std::invalid_argument(to_string(t.shape()) +
                                        " != " + to_string(shape_));
        }
        if (t.value_type() != value_type_) {
            // throw std::invalid_argument("type mismatch");
            throw std::invalid_argument(std::to_string(t.value_type()) +
                                        " != " + std::to_string(value_type_));
        }
        raw_tensor_ref<D> r((void *)t.data(), t.value_type(), t.shape());
        value_.emplace(r);
    }

    void unbind() { value_.reset(); }

    raw_tensor_ref<D> ref() const { return value_.value(); }

    raw_tensor_view<D> view() const
    {
        return raw_tensor_view<D>(value_.value());
    }
};

template <typename D>
class variable_manager
{
    std::vector<std::unique_ptr<variable<D>>> variables_;
    std::vector<std::unique_ptr<reference<D>>> references_;

  public:
    variable<D> *create_tensor(const tensor_symbol &sym)
    {
        variable<D> *v = new variable<D>(sym);
        variables_.emplace_back(v);
        return v;
    }

    reference<D> *create_tensor_reference(const tensor_symbol &sym)
    {
        reference<D> *r = new reference<D>(sym);
        references_.emplace_back(r);
        return r;
    }
};
}  // namespace ttl::nn::graph::internal
