INCLUDE(ExternalProject)

SET(STDTRACER_GIT_URL
    https://github.com/lgarithm/stdtracer.git
    CACHE STRING "URL for clone stdtracer")

SET(STDNN_OPS_GIT_URL
    https://github.com/lgarithm/stdnn-ops.git
    CACHE STRING "URL for clone stdnn-ops")

SET(STDNN_OPS_GIT_TAG "master" CACHE STRING "git tag for checkout stdnn-ops")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(stdnn-ops-repo
                    GIT_REPOSITORY ${STDNN_OPS_GIT_URL}
                    GIT_TAG ${STDNN_OPS_GIT_TAG}
                    PREFIX ${PREFIX}
                    # CONFIGURE_COMMAND
                    # ""
                    # BUILD_COMMAND
                    # ""
                    INSTALL_COMMAND ""
                    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX}
                               -DBUILD_TESTS=0
                               -DBUILD_EXAMPLES=0
                               -DBUILD_BENCHMARKS=0
                               -DBUILD_PACKAGES=0
                               -DUSE_OPENBLAS=0
                               # -DOpenBLAS_INCLUDE_DIRS=$ENV{HOME}/local/openbl
                               # as/include
                               # -DCMAKE_PREFIX_PATH=$ENV{HOME}/local/openblas
                               -DSTDTENSOR_GIT_URL=${STDTENSOR_GIT_URL}
                               -DSTDTENSOR_GIT_TAG=${STDTENSOR_GIT_TAG}
                               -DSTDTRACER_GIT_URL=${STDTRACER_GIT_URL})

INCLUDE_DIRECTORIES(${PREFIX}/src/stdnn-ops-repo/include)
# INCLUDE_DIRECTORIES(${PREFIX}/include)

FUNCTION(TARGET_USE_STDNN_OPS target)
    ADD_DEPENDENCIES(${target} stdnn-ops-repo)
ENDFUNCTION()
