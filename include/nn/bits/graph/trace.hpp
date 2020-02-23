#pragma once
#ifdef NN_GRAPH_ENABLE_TRACE
#    ifdef NN_GRAPH_ENABLE_CUDA
#        include <tracer/cuda>
#    else
#        include <tracer/simple>
#    endif
#else
#    include <tracer/disable>
#endif
