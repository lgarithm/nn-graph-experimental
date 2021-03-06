FUNCTION(ADD_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_USE_STDTRACER(${target})
    TARGET_USE_STDTENSOR(${target})
    TARGET_USE_STDNN_OPS(${target})
    IF(${ENABLE_CUDA})
        TARGET_LINK_LIBRARIES(${target} cudart)
        TARGET_USE_CUNN_OPS(${target})
    ENDIF()
    IF(USE_OPENBLAS)
        IF(APPLE)
            TARGET_LINK_LIBRARIES(${target} openblas)
        ELSEIF(BLAS_HOME)
            TARGET_LINK_LIBRARIES(${target} openblas)
        ELSE()
            TARGET_LINK_LIBRARIES(${target} ${OpenBLAS_LIBRARIES})
        ENDIF()
    ENDIF()
ENDFUNCTION()

FUNCTION(SIMPLE_EXAMPLE target)
    ADD_EXAMPLE(${target} examples/${target}.cpp)
ENDFUNCTION()

SIMPLE_EXAMPLE(a-plus-b)
SIMPLE_EXAMPLE(bench-ops)
SIMPLE_EXAMPLE(composite-derivative)
SIMPLE_EXAMPLE(mnist-dataset)
SIMPLE_EXAMPLE(pascal-triangle)
SIMPLE_EXAMPLE(quadratic-function)
SIMPLE_EXAMPLE(simple-derivative)
SIMPLE_EXAMPLE(train-mnist-cnn)
SIMPLE_EXAMPLE(train-mnist-slp)
SIMPLE_EXAMPLE(train-model)

OPTION(BUILD_GPU_EXAMPLES "Build gpu examples." OFF)

IF(BUILD_GPU_EXAMPLES)
    ENABLE_LANGUAGE(CUDA)
    SIMPLE_EXAMPLE(gpu-example)
ENDIF()
