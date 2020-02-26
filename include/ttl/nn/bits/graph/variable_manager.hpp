#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <ttl/debug>
#include <ttl/nn/bits/graph/common.hpp>
#include <ttl/nn/bits/graph/tensor.hpp>
#include <ttl/shape>
#include <ttl/tensor>

namespace ttl::nn::graph::internal
{
template <typename D>
class variable
{
  public:
    virtual ~variable() {}

    virtual raw_tensor_ref<D> raw_ref() const = 0;
    virtual raw_tensor_view<D> raw_view() const = 0;

    virtual size_t data_size() const = 0;

    virtual operator std::string() const = 0;
};

template <typename D>
class raw_tensor_variable : public variable<D>
{
    using E = typename raw_tensor<D>::encoder_type;
    using value_type_t = typename E::value_type;
    const raw_tensor<D> value_;

  public:
    raw_tensor_variable(const value_type_t &type, const flat_shape &shape)
        : value_(type, shape)
    {
    }

    raw_tensor_variable(const tensor_symbol &sym)
        : raw_tensor_variable(sym.value_type(), sym.shape())
    {
    }

    raw_tensor_ref<D> raw_ref() const override
    {
        return raw_tensor_ref<D>(value_);
    }

    raw_tensor_view<D> raw_view() const override
    {
        return raw_tensor_view<D>(value_);
    }

    size_t data_size() const override { return value_.data_size(); }

    operator std::string() const override
    {
        // return ttl::tensor_type_name(value_);
        return "T";
    }
};

template <typename D>
class reference
{
  public:
    virtual ~reference() {}

    virtual void bind(const raw_tensor_view<D> &) = 0;
    virtual void unbind() = 0;

    virtual raw_tensor_ref<D> raw_ref() const = 0;
    virtual raw_tensor_view<D> raw_view() const = 0;
};

template <typename D>
class raw_tensor_reference : public reference<D>
{
    using E = typename raw_tensor<D>::encoder_type;
    using value_type_t = typename E::value_type;

    const value_type_t value_type_;
    const flat_shape shape_;
    std::optional<raw_tensor_ref<D>> value_;

  public:
    raw_tensor_reference(const value_type_t &type, const flat_shape &shape)
        : value_type_(type), shape_(shape)
    {
    }

    raw_tensor_reference(const tensor_symbol &sym)
        : raw_tensor_reference(sym.value_type(), sym.shape())
    {
    }

    void bind(const raw_tensor_view<D> &t) override
    {
        if (t.shape().size() != shape_.size()) {
            throw std::invalid_argument(to_string(t.shape()) +
                                        " != " + to_string(shape_));
        }
        if (t.value_type() != value_type_) {
            throw std::invalid_argument("type mismatch");
            // throw std::invalid_argument(to_string(t.value_type()) +
            //                             " != " + to_string(value_type_));
        }
        raw_tensor_ref<D> r((void *)t.data(), t.value_type(), t.shape());
        value_.emplace(r);
    }

    void unbind() override { value_.reset(); }

    raw_tensor_ref<D> raw_ref() const override { return value_.value(); }

    raw_tensor_view<D> raw_view() const override
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
        variable<D> *v = new raw_tensor_variable<D>(sym);
        variables_.emplace_back(v);
        return v;
    }

    reference<D> *create_tensor_reference(const tensor_symbol &sym)
    {
        reference<D> *r = new raw_tensor_reference<D>(sym);
        references_.emplace_back(r);
        return r;
    }
};
}  // namespace ttl::nn::graph::internal
