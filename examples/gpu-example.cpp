#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

#include <nn/bits/ops/example_gpu_ops.hpp>

#include "trace.hpp"

void _eager_example()
{
    using R = float;
    const int n = 1 << 20;
    ttl::cuda_tensor<R, 1> x(n);
    ttl::cuda_tensor<R, 1> y(n);
    ttl::cuda_tensor<R, 1> z(n);
    example_ops::add()(ref(z), view(x), view(y));
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

    nn::graph::gpu_builder b;
    const int n = 1 << 20;

    auto x = b.covar<int>(b.shape(n));
    auto y = b.covar<int>(b.shape(n));
    auto z = b.invoke(example_ops::add(), x, y);

    nn::graph::gpu_runtime rt;
    b.build(rt);
    {
        example_ops::ones()(x->get_ref(rt));
        example_ops::ones()(y->get_ref(rt));
        example_ops::ones()(z->get_ref(rt));
    }
    b.run(rt, z);
    {
        auto v = z->get_view(rt);
        ttl::tensor<int, 1> zz(z->shape());
        v.to_host(zz.data());
        std::cerr << "z[0] = " << zz.data()[0] << std::endl;
    }
}

void gpu_sgd_example()
{
    LOG_SCOPE(__func__);

    nn::graph::gpu_builder b;

    auto x =
        b._covar<float, nn::graph::gpu>("x", b.shape());  //, nn::ops::ones());
    auto y = b.invoke(nn::ops::mul(), x, x);

    nn::graph::optimizer opt;
    auto train_step = opt.minimize<nn::graph::gpu>(b, y);

    nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    float e = 1;
    for (int i = 0; i < 10; ++i) {
        e *= 0.8;
        std::cerr << "step = " << i << ", 0.8 ^ " << i + 1 << " = " << e
                  << std::endl;
        b.run(rt, train_step);
        {
            // auto v = y->get_view(rt);
            // std::cerr << "y = " << v.data()[0] << std::endl;
        } {
            // auto v = x->get_view(rt);
            // std::cerr << "x = " << v.data()[0] << std::endl;
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
