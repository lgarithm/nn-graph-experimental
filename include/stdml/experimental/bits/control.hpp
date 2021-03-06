#pragma once
#include <ttl/cuda_tensor>
#include <ttl/nn/bits/graph/tensor.hpp>
#include <ttl/nn/bits/ops/blas.hpp>
#include <ttl/nn/bits/ops/init.hpp>
#include <ttl/range>
#include <ttl/tensor>

namespace stdml::internal
{
template <typename R, ttl::rank_t r, typename D>
void learn(const ttl::tensor_ref<R, r, D> &x,
           const ttl::tensor_view<R, r, D> &g, const R lr)
{
    ttl::tensor<R, 0, D> a;
    (ttl::nn::ops::constant<R>(-lr))(ttl::ref(a));
    ttl::nn::ops::axpy()(x, ttl::view(a), g, ttl::view(x));
}

#ifndef NN_GRAPH_ENABLE_CUDA
template <typename R, ttl::rank_t r>
void learn(const ttl::cuda_tensor_ref<R, r> &x,
           const ttl::cuda_tensor_view<R, r> &g, const R lr)
{
    printf("cuda not enabled!\n");
}
#endif

template <typename R /* explicit */, typename D, typename R1 /* auto */>
void learn(const ttl::nn::graph::internal::raw_tensor_ref<D> &x,
           const ttl::nn::graph::internal::raw_tensor_view<D> &g, const R1 lr)
{
    learn<R>(ttl::flatten(x.template typed<R>()),
             ttl::flatten(g.template typed<R>()), lr);
}

template <typename R /* explicit */, typename Pairs, typename RT,
          typename R1 /* auto */>
void learn_all(const Pairs &gvs, const RT &rt, const R1 lr)
{
    for (const auto &[g, v] : gvs) { learn<R>(rt.ref(v), rt.view(g), lr); }
}

template <typename F, typename... Args>
int batch_invoke(const int batch_size, const F &f, const Args &... args)
{
    const int n = std::get<0>(std::get<0>(std::make_tuple(args...)).dims());
    for (auto i : ttl::range(n / batch_size)) {
        f(args.slice(i * batch_size, (i + 1) * batch_size)...);
    }
    return n % batch_size;
}
}  // namespace stdml::internal
