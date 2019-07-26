INCLUDE(ExternalProject)

SET(CUNN_OPS_GIT_URL "" CACHE STRING "URL for clone cunn-ops")

SET(CUNN_OPS_GIT_TAG "master" CACHE STRING "git tag for checkout stdnn-ops")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(cunn-ops-repo
                    GIT_REPOSITORY ${CUNN_OPS_GIT_URL}
                    GIT_TAG ${CUNN_OPS_GIT_TAG}
                    PREFIX ${PREFIX}
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND ""
                    INSTALL_COMMAND "")

INCLUDE_DIRECTORIES(${PREFIX}/src/cunn-ops-repo/include)

SET(CUNN_OPS_SRC ${PREFIX}/src/cunn-ops-repo)
SET(SRC_PREFIX ${CUNN_OPS_SRC}/src)

ADD_LIBRARY(cunn-ops
            # BEGIN sort
            ${SRC_PREFIX}/nn/cuda/kernels/argmax.cu
            ${SRC_PREFIX}/nn/cuda/kernels/bias.cu
            ${SRC_PREFIX}/nn/cuda/kernels/col2im.cu
            ${SRC_PREFIX}/nn/cuda/kernels/contraction.cu
            ${SRC_PREFIX}/nn/cuda/kernels/im2col.cu
            ${SRC_PREFIX}/nn/cuda/kernels/mm.cu
            ${SRC_PREFIX}/nn/cuda/kernels/softmax.cu
            ${SRC_PREFIX}/nn/cuda/kernels/xentropy.cu
            ${SRC_PREFIX}/nn/cuda/ops/elementary.cu
            ${SRC_PREFIX}/nn/cuda/ops/init.cu
            ${SRC_PREFIX}/nn/cuda/ops/la.cpp
            ${SRC_PREFIX}/nn/cuda/ops/la.cu
            ${SRC_PREFIX}/nn/cuda/ops/reduce.cu
            ${SRC_PREFIX}/nn/cuda/ops/utility.cu
            # END sort
            )

FUNCTION(TARGET_USE_CUNN_OPS target)
    ADD_DEPENDENCIES(${target} cunn-ops-repo)
    TARGET_LINK_LIBRARIES(${target} cunn-ops)
ENDFUNCTION()
