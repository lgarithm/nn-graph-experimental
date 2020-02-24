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
template <typename R, rank_t r, typename D>
class tensor_variable;

template <typename R, rank_t r, typename D>
class tensor_reference;

template <typename D>
class variable
{
  public:
    virtual ~variable() {}

    virtual raw_tensor_ref<D> raw_ref() const = 0;
    virtual raw_tensor_view<D> raw_view() const = 0;

    template <typename R, rank_t r>
    tensor_variable<R, r, D> &as()
    {
        return *down_cast<tensor_variable<R, r, D>>(this);
    }

    virtual size_t data_size() const = 0;

    virtual operator std::string() const = 0;
};

template <typename R, rank_t r, typename D>
class tensor_variable : public variable<D>
{
    using T = ttl::tensor<R, r, D>;
    using Ref = ttl::tensor_ref<R, r, D>;

    T value_;

  public:
    tensor_variable(const ttl::shape<r> &shape) : value_(shape) {}

    Ref get() const { return Ref(value_); }

    raw_tensor_ref<D> raw_ref() const override
    {
        return raw_tensor_ref<D>(value_.data(), idx_encoder::value<R>(),
                                 value_.shape());
    }

    raw_tensor_view<D> raw_view() const override
    {
        return raw_tensor_view<D>(value_.data(), idx_encoder::value<R>(),
                                  value_.shape());
    }

    size_t data_size() const override { return value_.data_size(); }

    operator std::string() const override
    {
        return ttl::tensor_type_name(value_);
    }
};

template <typename D>
class reference
{
  public:
    virtual ~reference() {}

    virtual void unbind() = 0;

    virtual raw_tensor_ref<D> raw_ref() const = 0;
    virtual raw_tensor_view<D> raw_view() const = 0;

    template <typename R, rank_t r>
    tensor_reference<R, r, D> &as()
    {
        return *down_cast<tensor_reference<R, r, D>>(this);
    }
};

template <typename R, rank_t r, typename D>
class tensor_reference : public reference<D>
{
    using Ref = ttl::tensor_ref<R, r, D>;

    const ttl::shape<r> shape_;

    std::optional<Ref> value_;

  public:
    tensor_reference(const ttl::shape<r> &shape) : shape_(shape) {}

    void bind(const Ref &t)
    {
        if (t.shape() != shape_) {
            throw std::invalid_argument(to_string(t.shape()) +
                                        " != " + to_string(shape_));
        }
        value_.emplace(t);
    }

    void unbind() override { value_.reset(); }

    Ref get() const { return value_.value(); }

    raw_tensor_ref<D> raw_ref() const override
    {
        const auto &value = value_.value();
        return raw_tensor_ref<D>(value.data(), idx_encoder::value<R>(),
                                 value.shape());
    }

    raw_tensor_view<D> raw_view() const override
    {
        const auto &value = value_.value();
        return raw_tensor_view<D>(value.data(), idx_encoder::value<R>(),
                                  value.shape());
    }
};

template <typename D>
class variable_manager
{
    std::vector<std::unique_ptr<variable<D>>> variables_;
    std::vector<std::unique_ptr<reference<D>>> references_;

  public:
    template <typename R, rank_t r>
    tensor_variable<R, r, D> *create_tensor(const ttl::shape<r> &shape)
    {
        auto v = new tensor_variable<R, r, D>(shape);
        variables_.emplace_back(v);
        return v;
    }

    template <typename R, rank_t r>
    tensor_reference<R, r, D> *
    create_tensor_reference(const ttl::shape<r> &shape)
    {
        auto tr = new tensor_reference<R, r, D>(shape);
        references_.emplace_back(tr);
        return tr;
    }
};
}  // namespace ttl::nn::graph::internal
