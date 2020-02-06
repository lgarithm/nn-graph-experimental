#pragma once
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nn/bits/graph/common.hpp>
#include <ttl/shape>

namespace ttl::nn::graph::internal
{
template <typename R, ttl::rank_t r, typename D>
class tensor_variable;
template <typename R, ttl::rank_t r, typename D>
class tensor_reference;

class variable
{
  public:
    virtual ~variable() {}

    template <typename R, ttl::rank_t r, typename D>
    tensor_variable<R, r, D> &as()
    {
        return *down_cast<tensor_variable<R, r, D>>(this);
    }

    virtual size_t data_size() const = 0;

    virtual operator std::string() const = 0;
};

template <typename R, ttl::rank_t r, typename D>
class tensor_variable : public variable
{
    using T = ttl::tensor<R, r, D>;
    using Ref = ttl::tensor_ref<R, r, D>;

    T value_;

  public:
    tensor_variable(const ttl::shape<r> &shape) : value_(shape) {}

    Ref get() const { return Ref(value_); }

    size_t data_size() const override { return value_.data_size(); }

    operator std::string() const override
    {
        return ttl::tensor_type_name(value_);
    }
};

class reference
{
  public:
    virtual ~reference() {}

    virtual void unbind() = 0;

    template <typename R, ttl::rank_t r, typename D>
    tensor_reference<R, r, D> &as()
    {
        return *down_cast<tensor_reference<R, r, D>>(this);
    }
};

template <typename R, ttl::rank_t r, typename D>
class tensor_reference : public reference
{
    using Ref = ttl::tensor_ref<R, r, D>;

    const ttl::shape<r> shape_;

    std::optional<Ref> value_;

  public:
    tensor_reference(const ttl::shape<r> &shape) : shape_(shape) {}

    void bind(const Ref &t)
    {
        if (t.shape() != shape_) {
            throw std::logic_error("tensor_reference"
                                   "::"
                                   "bind"
                                   ": "
                                   " invalid shape");
        }
        value_.emplace(t);
    }

    void unbind() override { value_.reset(); }

    Ref get() const { return value_.value(); }
};

template <typename D>
class variable_manager
{
    std::vector<std::unique_ptr<variable>> variables_;
    std::vector<std::unique_ptr<reference>> references_;

  public:
    template <typename R, ttl::rank_t r>
    tensor_variable<R, r, D> *create_tensor(const ttl::shape<r> &shape)
    {
        auto v = new tensor_variable<R, r, D>(shape);
        variables_.push_back(std::unique_ptr<variable>(v));
        return v;
    }

    template <typename R, ttl::rank_t r>
    tensor_reference<R, r, D> *
    create_tensor_reference(const ttl::shape<r> &shape)
    {
        auto tr = new tensor_reference<R, r, D>(shape);
        references_.push_back(std::unique_ptr<reference>(tr));
        return tr;
    }
};
}  // namespace ttl::nn::graph::internal
