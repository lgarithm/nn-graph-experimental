INCLUDE(ExternalProject)
INCLUDE(cmake/deps.cmake)

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(
    stdnn-ops-cuda-repo
    GIT_REPOSITORY ${STDNN_OPS_CUDA_GIT_URL}
    GIT_TAG ${STDNN_OPS_CUDA_GIT_TAG}
    PREFIX ${PREFIX}
    CMAKE_ARGS -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
               -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
               -DCMAKE_BUILD_TYPE=Release
               -DTTL_HOME=${PREFIX}/src/stdtensor-repo
               -DNN_OPS_HOME=${PREFIX}/src/stdnn-ops-repo
               -DCMAKE_INSTALL_PREFIX=${PREFIX})

ADD_DEPENDENCIES(stdnn-ops-cuda-repo stdtensor-repo)
ADD_DEPENDENCIES(stdnn-ops-cuda-repo stdnn-ops-repo)
LINK_DIRECTORIES(${PREFIX}/lib)

FUNCTION(TARGET_USE_CUNN_OPS target)
    ADD_DEPENDENCIES(${target} stdnn-ops-cuda-repo)
    # TARGET_LINK_LIBRARIES(${target} stdnn-ops-cuda)
    TARGET_LINK_LIBRARIES(${target} stdnn-ops-cuda)
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${PREFIX}/include)
ENDFUNCTION()
