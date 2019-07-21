#pragma once
// #define USE_FAKE_CUDA_RUNTIME 1

#ifdef USE_FAKE_CUDA_RUNTIME
#include <ttl/bits/fake_cuda_runtime.hpp>
#else
#include <cuda_runtime.h>
#endif
