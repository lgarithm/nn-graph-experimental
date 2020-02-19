#include <nn/graph>
#include <ttl/nn/ops>
#include <ttl/tensor>

#include "utils.hpp"

void f(int n)
{
    TRACE_SCOPE("f");
    ttl::nn::graph::builder b;

    using V = ttl::nn::graph::internal::var_node<float, 0>;
    std::vector<std::vector<const V *>> vars;
    vars.resize(n + 1);
    for (auto i : ttl::range(n + 1)) { vars.at(i).resize(n + 1 - i); }

    for (auto i : ttl::range(n + 1)) {
        for (auto j [[maybe_unused]] : ttl::range(n + 1 - i)) {
            char name[32];
            sprintf(name, "x[%d][%d]", i, j);
            if (i == 0) {
                // fprintf(stderr, "var %s\n", name);
                vars.at(i).at(j) = b.covar<float>(name, ttl::make_shape(),
                                                  ttl::nn::ops::ones());
            } else {
                // fprintf(stderr, "fn %s\n", name);
                vars.at(i).at(j) = b.invoke<float>(
                    name, ttl::nn::ops::add(),  //
                    vars.at(i - 1).at(j), vars.at(i - 1).at(j + 1));
            }
        }
    }
    auto y = vars.at(n).at(0);
    auto gvs = b.gradients(y);
    printf("%d\n", (int)gvs.size());  // FIXME: merge gradients before return
}

int main()
{
    f(0);
    f(1);
    f(2);
    f(3);
    f(4);
    f(5);
    return 0;
}
