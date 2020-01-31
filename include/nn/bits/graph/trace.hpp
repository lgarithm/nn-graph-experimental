#pragma once
#ifdef NN_GRAPH_ENABLE_TRACE
#    include <nn/bits/graph/trace_enabled.hpp>
#else
#    define TRACE_CUDA_SCOPE(name)
#    define TRACE_SCOPE(name)
#    define LOG_SCOPE(name)
#    define TRACE_STMT(e) e;
#    define TRACE_EXPR(e) e
#endif
