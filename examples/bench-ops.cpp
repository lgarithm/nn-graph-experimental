#include "bench-utils.hpp"
#include <nn/graph>
#include <nn/ops>

void bench_sim()
{
    nn::graph::gpu_builder b;
    int n = 10000;
    auto x = b.var<int32_t>(b.shape(n));
    auto y = b.var<int32_t>(b.shape(n));
    auto acc = b.template invoke<float>(nn::ops::similarity(), x, y);
    nn::graph::gpu_runtime rt;
    b.build(rt);
    b.init(rt);

    ttl::cuda_tensor<int32_t, 1> x_data(n);
    ttl::cuda_tensor<int32_t, 1> y_data(n);
    for (auto i : ttl::range(60)) {
        TRACE_SCOPE("b::sim");
        rt.bind(x, ref(x_data));
        rt.bind(y, ref(y_data));
        b.run(rt, acc);
    }
}

void bench_ops(const std::string prefix)
{
    benchmark b(prefix);
    TRACE_SCOPE(__func__);
    {
        const int n = 10000;
        tt<float, 0> scalar_t;
        tt<int32_t, 1> seq_t(n);
        b(cpu(), nn::ops::similarity(), scalar_t, seq_t, seq_t);
        b(gpu(), nn::cuda::ops::similarity(), scalar_t, seq_t, seq_t);
    }
    {
        tt<float, 0> scalar_t;
        tt<float, 2> weight_t(28 * 28, 10);
        b(cpu(), nn::ops::axpy(), weight_t, scalar_t, weight_t, weight_t);
        b(gpu(), nn::cuda::ops::axpy(), weight_t, scalar_t, weight_t, weight_t);
    }
}

int main(int argc, char *argv[])
{
    TRACE_SCOPE(argv[0]);
    bench_sim();
    bench_ops("");
    return 0;
}
