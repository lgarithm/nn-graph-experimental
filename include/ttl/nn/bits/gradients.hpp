#pragma once
#include <ttl/nn/bits/ops/gradients/activation.hpp>
#include <ttl/nn/bits/ops/gradients/add.hpp>
#include <ttl/nn/bits/ops/gradients/bias.hpp>
#include <ttl/nn/bits/ops/gradients/conv2d.hpp>
#include <ttl/nn/bits/ops/gradients/matmul.hpp>
#include <ttl/nn/bits/ops/gradients/mul.hpp>
#include <ttl/nn/bits/ops/gradients/reshape.hpp>
#include <ttl/nn/bits/ops/gradients/softmax.hpp>
#include <ttl/nn/bits/ops/gradients/xentropy.hpp>

namespace ttl::nn
{
template <typename F, arity_t arity>
struct gradient {
    using type = void;
};

//
template <ttl::rank_t... rs>
struct gradient<ops::copy_flatten<rs...>, 0> {
    using type = ops::grad::copy_flatten<rs...>;
};

//
template <arity_t p>
struct gradient<ops::add, p> {
    using type = ops::grad::add<p>;
};

//
template <arity_t p>
struct gradient<ops::mul, p> {
    using type = ops::grad::mul<p>;
};

// enum class image_order = {ops::hw};

template <typename image_order, arity_t p>
struct gradient<ops::add_bias<image_order>, p> {
    using type = ops::grad::add_bias<image_order, p>;
};

//
template <>
struct gradient<ops::xentropy, 1> {
    using type = ops::grad::xentropy<1>;
};

//
template <>
struct gradient<ops::relu, 0> {
    using type = ops::grad::relu<0>;
};

template <>
struct gradient<ops::softmax, 0> {
    using type = ops::grad::softmax<0>;
};

//
template <arity_t p>
struct gradient<ops::matmul, p> {
    using type = ops::grad::matmul<p>;
};

//
template <typename image_order, typename filter_order, arity_t p>
struct gradient<ops::conv<image_order, filter_order>, p> {
    using type = ops::grad::conv<image_order, filter_order, p>;
};
}  // namespace ttl::nn
