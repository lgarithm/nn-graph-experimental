#pragma once

namespace nn::graph::internal
{

// A tensor_function wraps an operator in stdnn-ops or cunn-ops
template <typename Op> class tensor_function
{
    const Op f_;

  public:
    tensor_function(const Op &f) : f_(f) {}
};

}  // namespace nn::graph::internal
