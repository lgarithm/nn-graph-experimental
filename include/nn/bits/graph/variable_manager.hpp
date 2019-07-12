#pragma once
#include <optional>
#include <ttl/tensor>

namespace nn::graph::internal
{
template <typename R, ttl::rank_t r> class tensor_variable;

class variable
{
  public:
    virtual ~variable() {}

    template <typename R, ttl::rank_t r> tensor_variable<R, r> &as()
    {
        return *down_cast<tensor_variable<R, r>>(this);
    }
};

template <typename R, ttl::rank_t r> class tensor_variable : public variable
{
    ttl::tensor<R, r> value_;

  public:
    tensor_variable(const ttl::shape<r> &shape) : value_(shape) {}

    ttl::tensor_ref<R, r> get() const { return ref(value_); }
};

template <typename R, ttl::rank_t r> class tensor_reference;

class reference
{
  public:
    virtual ~reference() {}

    virtual void unbind() = 0;

    template <typename R, ttl::rank_t r> tensor_reference<R, r> &as()
    {
        return *down_cast<tensor_reference<R, r>>(this);
    }
};

template <typename R, ttl::rank_t r> class tensor_reference : public reference
{
    const ttl::shape<r> shape_;

    std::optional<ttl::tensor_ref<R, r>> value_;

  public:
    tensor_reference(const ttl::shape<r> &shape) : shape_(shape) {}

    void bind(const ttl::tensor_ref<R, r> &t)
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

    ttl::tensor_ref<R, r> get() const { return value_.value(); }
};

class variable_manager
{
    std::vector<std::unique_ptr<variable>> variables_;
    std::vector<std::unique_ptr<reference>> references_;

  public:
    template <typename R, ttl::rank_t r>
    tensor_variable<R, r> *create_tensor(const ttl::shape<r> &shape)
    {
        auto v = new tensor_variable<R, r>(shape);
        variables_.push_back(std::unique_ptr<variable>(v));
        return v;
    }

    template <typename R, ttl::rank_t r>
    tensor_reference<R, r> *create_tensor_reference(const ttl::shape<r> &shape)
    {
        auto tr = new tensor_reference<R, r>(shape);
        references_.push_back(std::unique_ptr<reference>(tr));
        return tr;
    }
};

}  // namespace nn::graph::internal
