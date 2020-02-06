#include "testing.hpp"

#include <nn/graph>

TEST(node_test, test1)
{
    using ttl::nn::graph::internal::forward_func_node;
    using ttl::nn::graph::internal::var_node;

    using X0 = var_node<float, 0>;
    using X1 = var_node<float, 0>;
    using F = ttl::nn::ops::add;
    using Y = var_node<float, 0>;

    X0 x0(ttl::make_shape(), "");  // FIXME: flip order
    X1 x1(ttl::make_shape(), "");
    Y y(ttl::make_shape(), "");

    forward_func_node<F, Y, X0, X1> f("f", F(), &y, &x0, &x1);

    constexpr auto arity = decltype(f)::arity;
    constexpr std::array<bool, arity> mask =
        f.is_differentiable(std::make_index_sequence<arity>());

    static_assert(std::get<0>(mask));
    static_assert(std::get<1>(mask));
}

class none_differentiable_binary_op
{
  public:
    template <ttl::rank_t r>
    ttl::shape<r> operator()(const ttl::shape<r> &x,
                             const ttl::shape<r> &y) const
    {
        return x;
    }

    template <typename R, ttl::rank_t r>
    void operator()(const ttl::tensor_ref<R, r> &z,
                    const ttl::tensor_view<R, r> &x,
                    const ttl::tensor_view<R, r> &y) const
    {
    }
};

TEST(node_test, test2)
{
    using ttl::nn::graph::internal::forward_func_node;
    using ttl::nn::graph::internal::var_node;

    using X0 = var_node<float, 0>;
    using X1 = var_node<float, 0>;
    using F = none_differentiable_binary_op;
    using Y = var_node<float, 0>;

    X0 x0(ttl::make_shape(), "");  // FIXME: flip order
    X1 x1(ttl::make_shape(), "");
    Y y(ttl::make_shape(), "");

    forward_func_node<F, Y, X0, X1> f("f", F(), &y, &x0, &x1);

    constexpr auto arity = decltype(f)::arity;
    const auto [a, b] = f.is_differentiable(std::make_index_sequence<arity>());

    ASSERT_TRUE(!a);
    ASSERT_TRUE(!b);
}
