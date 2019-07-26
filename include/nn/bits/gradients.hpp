#pragma once
#include <nn/bits/ops/constant.hpp>
#include <nn/experimental/bits/ops/grad/add.hpp>
#include <nn/experimental/bits/ops/grad/bias.hpp>
#include <nn/experimental/bits/ops/grad/conv.hpp>
#include <nn/experimental/bits/ops/grad/matmul.hpp>
#include <nn/experimental/bits/ops/grad/mul.hpp>
#include <nn/experimental/bits/ops/grad/reshape.hpp>
#include <nn/experimental/bits/ops/grad/softmax.hpp>
#include <nn/experimental/bits/ops/grad/xentropy.hpp>

namespace nn
{

template <typename F, int arity> struct gradient {
    using type = void;
};

//
template <ttl::rank_t r> struct gradient<nn::ops::reshape_copy<r>, 0> {
    using type = nn::experimental::ops::grad::reshape_copy<r>;
};

//
template <> struct gradient<nn::ops::add, 0> {
    using type = nn::experimental::ops::grad::add<0>;
};

template <> struct gradient<nn::ops::add, 1> {
    using type = nn::experimental::ops::grad::add<1>;
};

//
template <int p> struct gradient<nn::ops::mul, p> {
    using type = nn::experimental::ops::grad::mul<p>;
};

// enum class image_order = {ops::hw};

template <typename image_order, int p>
struct gradient<nn::ops::add_bias<image_order>, p> {
    using type = nn::experimental::ops::grad::add_bias<image_order, p>;
};

//
template <> struct gradient<nn::ops::xentropy, 1> {
    using type = nn::experimental::ops::grad::xentropy<1>;
};

//
template <> struct gradient<nn::ops::softmax, 0> {
    using type = nn::experimental::ops::grad::softmax<0>;
};

//
template <typename E, int p> struct gradient<nn::ops::matmul_<E>, p> {
    using type = nn::experimental::ops::grad::matmul<p, E>;
};

//
template <typename image_order, typename filter_order, int p>
struct gradient<nn::ops::conv<image_order, filter_order>, p> {
    using type =
        nn::experimental::ops::grad::conv<image_order, filter_order, p>;
};

}  // namespace nn
