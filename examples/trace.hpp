#include <stdtracer.hpp>

// #define TRACE_STACK

#ifdef TRACE_STACK
using tracer_t = scope_t_<stack_tracer_ctx_t>;

extern stack_tracer_ctx_t default_stack_tracer_ctx;

#define TRACE_SCOPE(name) tracer_t _((name), default_stack_tracer_ctx)

#define TRACE_STMT(e)                                                          \
    {                                                                          \
        tracer_t _(#e, default_stack_tracer_ctx);                              \
        e;                                                                     \
    }

#define TRACE_EXPR(e)                                                          \
    [&]() {                                                                    \
        tracer_t _(#e, default_stack_tracer_ctx);                              \
        return (e);                                                            \
    }()

#else

using tracer_t =
    multi_ctx_scope_t_<default_clock_t,
                       simple_tracer_ctx_t /* , log_tracer_ctx_t */>;

using logger_t =
    multi_ctx_scope_t_<default_clock_t, simple_tracer_ctx_t, log_tracer_ctx_t>;

extern simple_tracer_ctx_t default_simple_ctx;
extern log_tracer_ctx_t default_log_ctx;

#define LOG_SCOPE(name) logger_t _((name), default_simple_ctx, default_log_ctx)

#define TRACE_SCOPE(name)                                                      \
    tracer_t _((name), default_simple_ctx /*, default_log_ctx*/)

#define TRACE_STMT(e)                                                          \
    {                                                                          \
        tracer_t _(#e, default_simple_ctx /*, default_log_ctx*/);              \
        e;                                                                     \
    }

#define TRACE_EXPR(e)                                                          \
    [&]() {                                                                    \
        tracer_t _(#e, default_simple_ctx /*, default_log_ctx*/);              \
        return (e);                                                            \
    }()

#endif
