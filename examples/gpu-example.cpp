#include <nn/graph>
#include <nn/ops>
#include <ttl/tensor>

#include <nn/bits/ops/example_gpu_ops.hpp>

#include "trace.hpp"

void eager_example()
{
    LOG_SCOPE(__func__);

    using R = float;
    const int n = 1 << 20;
    ttl::cuda_tensor<R, 1> x(n);
    ttl::cuda_tensor<R, 1> y(n);
    ttl::cuda_tensor<R, 1> z(n);

    example_ops::add()(ref(z), view(x), view(y));
}

void graph_example()
{
    LOG_SCOPE(__func__);

    nn::graph::builder b;

    auto x = b.covar<int>(ttl::make_shape());
    auto y = b.covar<int>(ttl::make_shape());
    auto z = b.invoke(nn::ops::add(), x, y);

    nn::graph::runtime rt;
    b.build(rt);
    {
        ttl::fill(x->get_ref(rt), 1);
        ttl::fill(y->get_ref(rt), 2);
    }
    b.run(rt, z);
    {
        auto v = z->get_view(rt);
        std::cerr << "z[0] = " << v.data()[0] << std::endl;
    }

    // TODO: gpu runtime example
}

int main(int argc, char *argv[])
{
    LOG_SCOPE(argv[0]);
    eager_example();
    graph_example();
    return 0;
}
