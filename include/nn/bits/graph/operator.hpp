#pragma once

namespace nn::graph::internal
{

// A tensor_function wraps an operator in stdnn-ops or stdnn-ops-cuda
template <typename Op> class tensor_function
{
    const Op f_;

  public:
    tensor_function(const Op &f) : f_(f) {}
};

template <bool, class F, class B> struct create_op_t;

template <class F, class B> struct create_op_t<false, F, B> {
    F operator()(const B &b) const
    {
        F f;
        return f;
    }
};

template <class F, class B> struct create_op_t<true, F, B> {
    F operator()(const B &b) const { return F(b); }
};

template <typename F, typename B> F create_op(const B &b)
{
    return create_op_t<std::is_constructible<F, B>::value, F, B>()(b);
}

#ifdef NN_GRAPH_ENABLE_CUDA
template <typename F, typename R>
cuda::ops::constant<R> create_op(const nn::ops::constant<R> &f)
{
    ttl::tensor<R, 0> value;
    f(ref(value));
    return cuda::ops::constant<R>(value.data()[0]);
}
#endif
}  // namespace nn::graph::internal
