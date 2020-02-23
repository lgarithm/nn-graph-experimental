// #define NN_GRAPG_TRACE 1

#include <ttl/experimental/copy>
#include <ttl/nn/computation_graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

template <typename T>
auto make_tensor_like(const T &t)
{
    return ttl::tensor<typename T::value_type, T::rank>(t.shape());
}

void _eager_example()
{
    using R = float;
    const int n = 1 << 20;
    ttl::cuda_tensor<R, 1> x(n);
    ttl::cuda_tensor<R, 1> y(n);
    ttl::cuda_tensor<R, 1> z(n);
    ttl::nn::ops::add()(ref(z), view(x), view(y));
}

void warmup()
{
    LOG_SCOPE(__func__);
    _eager_example();
}

void eager_example()
{
    LOG_SCOPE(__func__);
    _eager_example();
}

void graph_example()
{
    LOG_SCOPE(__func__);

    ttl::nn::graph::gpu_builder b;
    const int n = 1 << 20;

    auto x = b.covar<int>(b.shape(n));
    auto y = b.covar<int>(b.shape(n));
    auto z = b.invoke(ttl::nn::ops::add(), x, y);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    {
        ttl::nn::ops::ones()(x->get_ref(rt));
        ttl::nn::ops::ones()(y->get_ref(rt));
        ttl::nn::ops::ones()(z->get_ref(rt));
    }
    b.run(rt, z);
    {
        auto v = z->get_view(rt);
        ttl::tensor<int, 1> zz(z->shape());
        ttl::copy(ttl::ref(zz), v);
        std::cerr << "z[0] = " << zz.data()[0] << std::endl;
    }
}

void gpu_sgd_example()
{
    LOG_SCOPE(__func__);

    ttl::nn::graph::gpu_builder b;

    auto x = b.covar<float>("x", b.shape(), ttl::nn::ops::ones());
    auto y = b.invoke(ttl::nn::ops::mul(), x, x);

    ttl::nn::graph::optimizer opt;
    auto train_step = opt.minimize(b, y);

    ttl::nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        std::cerr << "step = " << i << ", 0.8 ^ " << i + 1 << " = " << e
                  << std::endl;
        b.run(rt, train_step);
        {
            auto v = y->get_view(rt);
            auto vv = make_tensor_like(v);
            ttl::copy(ttl::ref(vv), v);
            std::cerr << "y = " << vv.data()[0] << std::endl;
        }
        {
            auto v = x->get_view(rt);
            auto vv = make_tensor_like(v);
            ttl::copy(ttl::ref(vv), v);
            std::cerr << "x = " << vv.data()[0] << std::endl;
        }
    }
}

int main(int argc, char *argv[])
{
    LOG_SCOPE(argv[0]);
    warmup();
    eager_example();
    graph_example();
    gpu_sgd_example();
    return 0;
}
