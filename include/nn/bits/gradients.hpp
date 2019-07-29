#pragma once
#include <nn/bits/ops/constant.hpp>
#include <nn/bits/ops/gradients/add.hpp>
#include <nn/bits/ops/gradients/bias.hpp>
#include <nn/bits/ops/gradients/conv.hpp>
#include <nn/bits/ops/gradients/matmul.hpp>
#include <nn/bits/ops/gradients/mul.hpp>
#include <nn/bits/ops/gradients/reshape.hpp>
#include <nn/bits/ops/gradients/softmax.hpp>
#include <nn/bits/ops/gradients/xentropy.hpp>

namespace nn
{

template <typename F, int arity> struct gradient {
    using type = void;
};

//
template <ttl::rank_t r> struct gradient<ops::reshape_copy<r>, 0> {
    using type = ops::grad::reshape_copy<r>;
};

//
template <> struct gradient<ops::add, 0> {
    using type = ops::grad::add<0>;
};

template <> struct gradient<ops::add, 1> {
    using type = ops::grad::add<1>;
};

//
template <int p> struct gradient<ops::mul, p> {
    using type = ops::grad::mul<p>;
};

// enum class image_order = {ops::hw};

template <typename image_order, int p>
struct gradient<ops::add_bias<image_order>, p> {
    using type = ops::grad::add_bias<image_order, p>;
};

//
template <> struct gradient<ops::xentropy, 1> {
    using type = ops::grad::xentropy<1>;
};

//
template <> struct gradient<ops::softmax, 0> {
    using type = ops::grad::softmax<0>;
};

//
template <typename E, int p> struct gradient<ops::matmul_<E>, p> {
    using type = ops::grad::matmul<p, E>;
};

//
template <typename image_order, typename filter_order, int p>
struct gradient<ops::conv<image_order, filter_order>, p> {
    using type = ops::grad::conv<image_order, filter_order, p>;
};

}  // namespace nn
